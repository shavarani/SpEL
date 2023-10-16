"""
The implementation for general knowledge fine-tuning, step one, which uses DistributedDataParallel for multi-GPU processing.
 The source code in here has been developed on the basis of explanations and the code provided in:
  https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html#diff-for-single-gpu-py-v-s-multigpu-py

Note: this is the most time-consuming step of the project. Make sure you provide more than one GPU to the script.

Running this script will automatically download enwiki-2023-spel-roberta-tokenized-aug-27-2023.tar.gz (19.1 GBs)
 into /home/<user_name>/.cache/torch/text/datasets/ (in linux systems) and will cache the validation set in this dataset
 into .checkpoints named with the format: validation_data_cache_b_<batch_size>_l_<label_size>_wiki.
 You do not need to worry about downloading or preprocessing the fine-tuning data.
 As well, the big tar.gz data file will not be extracted on your disc, so you do not need to reserve more than 18.9 GBs
  on disk to store the fine-tuning dataset.
"""
import math
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from spel.model import SpELAnnotator
from spel.data_loader import get_dataset
from spel.utils import store_validation_data_wiki
from spel.configuration import get_checkpoints_dir


TRACK_WITH_WANDB = True
if TRACK_WITH_WANDB:
    try:
        import wandb
    except ModuleNotFoundError:
        TRACK_WITH_WANDB = False

ENABLE_AMP = False  # enable mixed precision gradient computation


def ddp_setup(rank, _world_size_):
    """
    Args:
        rank: Unique identifier of each process
        _world_size_: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="gloo", rank=rank, world_size=_world_size_)


def get_mode_checkpoint_name():
    return os.path.join(get_checkpoints_dir(), "spel-step-1-ddp")


def finetune_step_1(rank: int, world_size: int, n_epochs, batch_size, eval_batch_size=32, encoder_lr=5e-5,
                    decoder_lr=0.01, accumulate_batch_gradients=4, label_size=8196, evaluate_every=15000,
                    n_epochs_freeze_bert=3, early_stopping_n_epoch_wait=2,
                    continue_from_previous_checkpoint=False, optimizer_warmup_steps=0):
    device = torch.device('cuda', rank)
    ddp_setup(rank, world_size)
    model = SpELAnnotator()
    model.get_mode_checkpoint_name = get_mode_checkpoint_name
    model.init_model_from_scratch(device=device)
    if continue_from_previous_checkpoint:
        model.load_checkpoint('spel-step-1-ddp.pt', device=device, rank=rank)
    if TRACK_WITH_WANDB and rank == 0:
        wandb.init(
            project="spel-finetune-step-1-ddp",
            config={
                "n_epochs": n_epochs,
                "batch_size": batch_size,
                "eval_batch_size": eval_batch_size,
                "encoder_lr": encoder_lr,
                "decoder_lr": decoder_lr,
                "accumulate_batch_gradients": accumulate_batch_gradients,
                "label_size": label_size,
                "evaluate_every": evaluate_every,
                "n_epochs_freeze_bert": n_epochs_freeze_bert,
                "early_stopping_n_epoch_wait": early_stopping_n_epoch_wait,
                "continue_from_previous_checkpoint": continue_from_previous_checkpoint
            }
        )
    original_optimizers = model.create_optimizers(encoder_lr, decoder_lr)
    original_warmup_schedulers = [model.create_warmup_scheduler(o, optimizer_warmup_steps)
                                  for o in original_optimizers] if optimizer_warmup_steps > 0 else None
    criterion = nn.BCEWithLogitsLoss()
    best_f1 = 0.0
    best_f1_per_epoch = [0.0 for _ in range(n_epochs)]

    model.bert_lm = DDP(model.bert_lm.to(device), device_ids=[rank], output_device=rank)
    model.out = DDP(model.out.to(device), device_ids=[rank], output_device=rank)

    fp16_scaler = torch.cuda.amp.GradScaler(enabled=True) if ENABLE_AMP else None

    for epoch in range(n_epochs):
        if epoch > 0:
            # we think of each past epoch as one initialization step that helps the optimization process start in a
            # better place in space
            model.bert_lm = model.bert_lm.module
            model.out = model.out.module
            model.load_checkpoint('spel-step-1-ddp.pt', device=device, rank=rank)
            model.bert_lm = DDP(model.bert_lm.to(device), device_ids=[rank], output_device=rank)
            model.out = DDP(model.out.to(device), device_ids=[rank], output_device=rank)
        if rank == 0:
            print(f"Beginning finetune (step 1) epoch {epoch} ...")
        # ######### freezing the RoBERTa parameters in the first `n_epochs_freeze_bert` epochs in fine-tuning ######
        if epoch < n_epochs_freeze_bert:
            optimizers = [original_optimizers[-1]]
            warmup_schedulers = [original_warmup_schedulers[-1]] if original_warmup_schedulers else None
        else:
            optimizers = original_optimizers
            warmup_schedulers = original_warmup_schedulers if original_warmup_schedulers else None
        for optimizer in optimizers:
            optimizer.zero_grad()
        # ##########################################################################################################
        # ###### Early Stopping code which will break the fine-tuning loop after `early_stopping_n_epoch_wait` epochs
        # ###### with no improvement
        if 0 < early_stopping_n_epoch_wait < epoch:
            should_early_stop = True
            ref_idx = epoch - early_stopping_n_epoch_wait - 1
            for i in range(1, early_stopping_n_epoch_wait + 1):
                if best_f1_per_epoch[ref_idx] < best_f1_per_epoch[ref_idx+i]:
                    should_early_stop = False
                    break
            if should_early_stop:
                print(f"Early stopping triggered, ending fine-tuning after {epoch} epochs, "
                      f"the sequence of best F-1 scores per epoch: {best_f1_per_epoch[:epoch]}")
                break
        # #########################################################################################################
        created_dataset = get_dataset(
            dataset_name='enwiki', get_labels_with_high_model_score=model.get_highest_confidence_model_predictions,
            split='train', batch_size=batch_size, label_size=label_size, load_distributed=True, world_size=world_size,
            rank=rank)
        created_dataset.sampler.set_epoch(epoch)
        _iter_ = tqdm(enumerate(created_dataset), disable=rank != 0)
        total_loss = 0
        cnt_loss = 0
        for iter_, (inputs, subword_mentions) in _iter_:
            subword_mentions_probs = subword_mentions.probs.to(device)
            if ENABLE_AMP:
                with torch.cuda.amp.autocast():
                    logits = model.get_model_raw_logits_training(
                        inputs.token_ids.to(device), subword_mentions.ids.to(device), subword_mentions_probs)
                    logits = logits.view(-1)
                    label_probs = subword_mentions_probs.view(-1)
                    loss = criterion(logits, label_probs)
            else:
                logits = model.get_model_raw_logits_training(
                    inputs.token_ids.to(device), subword_mentions.ids.to(device), subword_mentions_probs)
                logits = logits.view(-1)
                label_probs = subword_mentions_probs.view(-1)
                loss = criterion(logits, label_probs)
            total_loss += loss.detach().item()
            cnt_loss += 1.0
            if ENABLE_AMP:
                fp16_scaler.scale(loss).backward()
            else:
                loss.backward()

            if (iter_ + 1) % accumulate_batch_gradients == 0:
                for optimizer in optimizers:
                    if ENABLE_AMP:
                        fp16_scaler.step(optimizer)
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                if ENABLE_AMP:
                    fp16_scaler.update()
                if warmup_schedulers:
                    for ws in warmup_schedulers:
                        ws.step()

            if (iter_ + 1) % evaluate_every == 0 and rank == 0:
                del inputs, subword_mentions, logits, label_probs, loss
                print("Evaluating the model ...")
                precision, recall, f1, f05, num_proposed, num_correct, num_gold, subword_eval = model.evaluate(
                    epoch, eval_batch_size, label_size, best_f1, is_training=True, potent_score_threshold=0.82)
                if best_f1 < f1:
                    best_f1 = f1
                    best_f1_per_epoch[epoch] = f1
                print(f"Evaluation results: precision={precision:.5f}, recall={recall:.5f}, f1={f1:.5f}, "
                      f"f05={f05:.5f}, num_proposed={num_proposed}, num_correct={num_correct}, num_gold={num_gold}")
                mm_alloc = sum([torch.cuda.memory_allocated(device=torch.cuda.device(i)) / (math.pow(2, 30))
                                for i in range(torch.cuda.device_count())])
                print(f"Current allocated memory: {mm_alloc:.4f} GB\n")
                if TRACK_WITH_WANDB:
                    wandb.log({
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "f05": f05,
                        "num_proposed": num_proposed,
                        "num_correct": num_correct,
                        "num_gold": num_gold,
                        "epoch": epoch,
                        "instance_number": iter_,
                        "allocated_memory": mm_alloc
                    })
            _iter_.set_description(f"Avg Loss: {total_loss/cnt_loss:.7f}")
            if TRACK_WITH_WANDB and rank == 0 and iter_ % 250 == 0 and iter_ > 0:
                wandb.log({"avg_loss": total_loss/cnt_loss})
        if rank == 0:
            print(f"\nEvaluating at the end of epoch {epoch}")
            precision, recall, f1, f05, num_proposed, num_correct, num_gold, subword_eval = model.evaluate(
                epoch, eval_batch_size, label_size, best_f1, is_training=True, potent_score_threshold=0.82)
            if best_f1 < f1:
                best_f1 = f1
            print(f"Evaluation results: precision={precision:.5f}, recall={recall:.5f}, f1={f1:.5f}, f05={f05:.5f}, "
                  f"num_proposed={num_proposed}, num_correct={num_correct}, num_gold={num_gold}")
            mm_alloc = sum([torch.cuda.memory_allocated(device=torch.cuda.device(i)) / (math.pow(2, 30))
                            for i in range(torch.cuda.device_count())])
            print(f"Current allocated memory: {mm_alloc:.4f} GB\n")
            if TRACK_WITH_WANDB:
                wandb.log({
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "f05": f05,
                    "num_proposed": num_proposed,
                    "num_correct": num_correct,
                    "num_gold": num_gold,
                    "epoch": epoch,
                    "allocated_memory": mm_alloc
                })
    destroy_process_group()


if __name__ == "__main__":
    try:
        w_size = torch.cuda.device_count()
        _arguments_ = {
            "world_size": w_size,
            "n_epochs": 14,
            "batch_size": 10,
            "eval_batch_size": 4,
            "encoder_lr": 5e-5,
            "decoder_lr": 0.01,
            "accumulate_batch_gradients": 4,
            "label_size": 10240,
            "evaluate_every": int(4000/w_size) if w_size > 1 else 15000,
            "n_epochs_freeze_bert": 3,
            "early_stopping_n_epoch_wait": 2,
            "continue_from_previous_checkpoint": False,
            "optimizer_warmup_steps": 0
        }
        args = (_arguments_["world_size"], _arguments_["n_epochs"], _arguments_["batch_size"],
                _arguments_["eval_batch_size"], _arguments_["encoder_lr"],
                _arguments_["decoder_lr"], _arguments_["accumulate_batch_gradients"],
                _arguments_["label_size"], _arguments_["evaluate_every"],
                _arguments_["n_epochs_freeze_bert"], _arguments_["early_stopping_n_epoch_wait"],
                _arguments_["continue_from_previous_checkpoint"],
                _arguments_["optimizer_warmup_steps"])
        # We need the following two lines to make sure the validation dataloader is not opened up while the train
        # dataloder is in use which will lead to the code slow-down (or sometimes freezing).
        print("Assuring validation data is cached before starting the training processes ...")
        store_validation_data_wiki(
            get_checkpoints_dir(), _arguments_["eval_batch_size"], _arguments_["label_size"],
            is_training=True, use_retokenized_wikipedia_data=False)
        mp.spawn(finetune_step_1, args=args, nprocs=w_size)
    finally:
        if TRACK_WITH_WANDB:
            wandb.finish()
