"""
The implementation for general knowledge fine-tuning, step two, which can use nn.DataParallel for multi-GPU processing.

Running this script will automatically download enwiki-2023-spel-roberta-tokenized-aug-27-2023-retokenized.tar.gz (17.5 GBs)
 into /home/<user_name>/.cache/torch/text/datasets/ (in linux systems). The validation set in this dataset will be cached
 the first time the evaluate function is called and the cached data will be stored into .checkpoints named with the
 format: validation_data_cache_b_<batch_size>_l_<label_size>_rt_wiki.
 You do not need to worry about downloading or preprocessing the fine-tuning data.
 As well, the big tar.gz data file will not be extracted on your disc, so you do not need to reserve more than 17.3 GBs
  on disk to store this fine-tuning dataset.
"""
import math
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from spel.model import SpELAnnotator
from spel.data_loader import get_dataset
from spel.configuration import device

TRACK_WITH_WANDB = True
if TRACK_WITH_WANDB:
    try:
        import wandb
    except ModuleNotFoundError:
        TRACK_WITH_WANDB = False

FINETUNE_MULTI_GPU = False

def symmetric_KL_loss(_input_, target, reduction='batchmean'):
    """ symmetric KL-divergence 1/2*(KL(p||q)+KL(q||p)) """
    _input_ = _input_.float()
    target = target.float()
    loss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(_input_, dim=-1, dtype=torch.float32),
                                      torch.nn.functional.softmax(target.detach(), dim=-1, dtype=torch.float32), reduction=reduction) + \
           torch.nn.functional.kl_div(torch.nn.functional.log_softmax(target, dim=-1, dtype=torch.float32),
                                      torch.nn.functional.softmax(_input_.detach(), dim=-1, dtype=torch.float32), reduction=reduction)
    return 0.5 * loss.sum()


class FinetuneS2(SpELAnnotator):
    def __init__(self):
        super(FinetuneS2, self).__init__()

    def finetune(self, checkpoint_name, n_epochs, batch_size, eval_batch_size=32, encoder_lr=5e-5, decoder_lr=0.01,
                 accumulate_batch_gradients=4, label_size=8196, evaluate_every=15000, n_epochs_freeze_bert=3,
                 early_stopping_n_epoch_wait=2, optimizer_warmup_steps=0, enable_r_drop=False, bert_dropout=0):
        self.init_model_from_scratch(device=device)
        if checkpoint_name is None:
            self.load_checkpoint(None, device=device, load_from_torch_hub=True, finetuned_after_step=1)
            checkpoint_name = 'enwiki_finetuned_step_1_model_checkpoint'
        else:
            self.load_checkpoint(checkpoint_name, device=device)
        if TRACK_WITH_WANDB:
            wandb.init(
                project="spel-finetune-step-2",
                config={
                    "checkpoint_name": checkpoint_name,
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
                    "optimizer_warmup_steps": optimizer_warmup_steps,
                    "enable_r_drop": enable_r_drop,
                    "bert_dropout": bert_dropout
                }
            )
        if bert_dropout > 0:
            for m in self.bert_lm.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.p = bert_dropout
        elif bert_dropout == 0:
            enable_r_drop = False
        original_optimizers = self.create_optimizers(encoder_lr, decoder_lr)
        original_warmup_schedulers = [self.create_warmup_scheduler(o, optimizer_warmup_steps)
                                      for o in original_optimizers] if optimizer_warmup_steps > 0 else None
        criterion = nn.BCEWithLogitsLoss()
        best_f1 = 0.0
        best_f1_per_epoch = [0.0 for _ in range(n_epochs)]
        if FINETUNE_MULTI_GPU:
            self.bert_lm = nn.DataParallel(self.bert_lm).to(device)
            self.out = nn.DataParallel(self.out).to(device)
        for epoch in range(n_epochs):
            print(f"Beginning fine-tune epoch {epoch} ...")
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
            # ##### Early Stopping code which will break the fine-tuning loop after `early_stopping_n_epoch_wait` epochs
            # ##### with no improvement
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
            _iter_ = tqdm(enumerate(
                get_dataset(dataset_name='enwiki', split='train', batch_size=batch_size, label_size=label_size,
                            get_labels_with_high_model_score=self.get_highest_confidence_model_predictions,
                            use_retokenized_wikipedia_data=True)
            ))
            total_loss = 0
            cnt_loss = 0
            for iter_, (inputs, subword_mentions) in _iter_:
                # inputs.eval_mask, subword_mentions.dictionary, inputs.raw_mentions are not used!
                subword_mentions_probs = subword_mentions.probs.to(device)
                logits = self.get_model_raw_logits_training(
                    inputs.token_ids.to(device), subword_mentions.ids.to(device), subword_mentions_probs)
                logits = logits.view(-1)
                label_probs = subword_mentions_probs.view(-1)

                loss = criterion(logits, label_probs)
                if enable_r_drop:
                    logits_additional = self.get_model_raw_logits_training(
                        inputs.token_ids.to(device), subword_mentions.ids.to(device), subword_mentions_probs)
                    logits_additional = logits_additional.view(-1)
                    loss += symmetric_KL_loss(logits, logits_additional)
                total_loss += loss.detach().item()
                cnt_loss += 1.0
                loss.backward()

                if (iter_ + 1) % accumulate_batch_gradients == 0:
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    if warmup_schedulers:
                        for ws in warmup_schedulers:
                            ws.step()

                if (iter_ + 1) % evaluate_every == 0:
                    del inputs, subword_mentions, logits, label_probs, loss, subword_mentions_probs
                    print("Evaluating the model ...")
                    precision, recall, f1, f05, num_proposed, num_correct, num_gold, subword_eval = self.evaluate(
                        epoch, eval_batch_size, label_size, best_f1, is_training=True,
                        use_retokenized_wikipedia_data=True, potent_score_threshold=0.77)
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
                if TRACK_WITH_WANDB and iter_ % 500 == 499:
                    wandb.log({"avg_loss": total_loss/cnt_loss})

            print(f"\nEvaluating at the end of epoch {epoch}")
            precision, recall, f1, f05, num_proposed, num_correct, num_gold, subword_eval = self.evaluate(
                epoch, eval_batch_size, label_size, best_f1, is_training=True, use_retokenized_wikipedia_data=True,
                potent_score_threshold=0.77)
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

    def get_mode_checkpoint_name(self):
        return os.path.join(self.checkpoints_root, "spel-step-2")


if __name__ == '__main__':
    try:
        b_annotator = FinetuneS2()
        b_annotator.finetune(checkpoint_name=None, n_epochs=14, batch_size=10, eval_batch_size=4, label_size=10240,
                             n_epochs_freeze_bert=3, optimizer_warmup_steps=0, early_stopping_n_epoch_wait=2,
                             enable_r_drop=False, bert_dropout=0)
    finally:
        if TRACK_WITH_WANDB:
            wandb.finish()
