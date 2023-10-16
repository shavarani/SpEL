"""
The implementation for domain specific fine-tuning, step three. This process is very light and can run on one Nvidia
1060 with 6 GBs of GPU memory.

Running this script will automatically download aida-conll-spel-roberta-tokenized-aug-23-2023.tar.gz (5.1 MBs)
 into /home/<user_name>/.cache/torch/text/datasets/ (in linux systems). The validation set in this dataset will be cached
 the first time the evaluate function is called and the cached data will be stored into .checkpoints named with the
 format: validation_data_cache_b_<batch_size>_l_<label_size>_conll. You do not need to worry about downloading or
 preprocessing the fine-tuning data. The tar.gz data file will not be extracted on your disc.
"""
import os
import math
import torch
import torch.nn as nn
from tqdm import tqdm

from spel.model import SpELAnnotator
from spel.data_loader import get_dataset
from spel.configuration import device

TRACK_WITH_WANDB = True
if TRACK_WITH_WANDB:
    import wandb


class FinetuneS3(SpELAnnotator):
    def __init__(self):
        super(FinetuneS3, self).__init__()

    def finetune(self, checkpoint_name, n_epochs, batch_size, bert_dropout=0.2, encoder_lr=5e-5, label_size=8196,
                 accumulate_batch_gradients=4,  exclude_parameter_names_regex='embeddings|encoder\\.layer\\.[0-2]\\.',
                 eval_batch_size=1):
        self.init_model_from_scratch(device=device)
        if checkpoint_name is None:
            self.load_checkpoint(None, device=device, load_from_torch_hub=True, finetuned_after_step=2)
            checkpoint_name = 'enwiki_finetuned_step_2_model_checkpoint'
        else:
            self.load_checkpoint(checkpoint_name, device=device)
        self.shrink_classification_head_to_aida(device=device)
        if label_size > self.out_module.num_embeddings:
            label_size = self.out_module.num_embeddings
        if TRACK_WITH_WANDB:
            wandb.init(
                project="spel-finetune-step-3",
                config={
                    "checkpoint_name": checkpoint_name,
                    "n_epochs": n_epochs,
                    "batch_size": batch_size,
                    "bert_dropout": bert_dropout,
                    "encoder_lr": encoder_lr,
                    "label_size": label_size,
                    "accumulate_batch_gradients": accumulate_batch_gradients,
                    "exclude_parameter_names_regex": exclude_parameter_names_regex,
                    "eval_batch_size": eval_batch_size
                }
            )
        self.bert_lm.train()
        self.out.train()
        if bert_dropout > 0:
            for m in self.bert_lm.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.p = bert_dropout
        optimizers = self.create_optimizers(encoder_lr, 0.0, exclude_parameter_names_regex)
        criterion = nn.BCEWithLogitsLoss()
        best_f1 = 0.0
        for epoch in range(n_epochs):
            print(f"Beginning fine-tune epoch {epoch} ...")
            _iter_ = tqdm(enumerate(
                get_dataset(dataset_name='aida', split='train', batch_size=batch_size, label_size=label_size,
                            get_labels_with_high_model_score=self.get_highest_confidence_model_predictions)
            ))
            total_loss = 0
            cnt_loss = 0
            for iter_, (inputs, subword_mentions) in _iter_:
                # inputs.eval_mask, subword_mentions.dictionary, inputs.raw_mentions are not used!
                subword_mentions_probs = subword_mentions.probs.to(device)
                logits = self.get_model_raw_logits_training(
                    inputs.token_ids.to(device), subword_mentions.ids.to(device), subword_mentions_probs)
                logits = logits.view(-1)  # (N*T, VOCAB)
                label_probs = subword_mentions_probs.view(-1)  # (N*T,)

                loss = criterion(logits, label_probs)
                total_loss += loss.detach().item()
                cnt_loss += 1.0
                loss.backward()

                if (iter_ + 1) % accumulate_batch_gradients == 0:
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()

                del logits
                del loss
                del inputs, subword_mentions, label_probs, subword_mentions_probs
                _iter_.set_description(f"Avg Loss: {total_loss/cnt_loss:.7f}")
                if TRACK_WITH_WANDB and iter_ % 50 == 49:
                    wandb.log({"avg_loss": total_loss/cnt_loss})
            print(f"\nEvaluating at the end of epoch {epoch}")

            sprecision, srecall, sf1, sf05, snum_proposed, snum_correct, snum_gold, subword_eval = self.evaluate(
                epoch, eval_batch_size, label_size, 1.1, is_training=False)
            inference_results = self.inference_evaluate(epoch, best_f1)
            if1 = inference_results.micro_entity_linking.f1.compute()
            if best_f1 < if1:
                best_f1 = if1
            print(f"Subword-level evaluation results: precision={sprecision:.5f}, recall={srecall:.5f}, f1={sf1:.5f}, "
                  f"f05={sf05:.5f}, num_proposed={snum_proposed}, num_correct={snum_correct}, num_gold={snum_gold}")
            print("Entity-level evaluation results:")
            print(inference_results)
            mm_alloc = torch.cuda.memory_allocated() / (math.pow(2, 30))
            print(f"Current allocated memory: {mm_alloc:.4f} GB")
            if TRACK_WITH_WANDB:
                wandb.log({
                    "s_precision": sprecision,
                    "s_recall": srecall,
                    "s_f1": sf1,
                    "s_f05": sf05,
                    "s_num_proposed": snum_proposed,
                    "s_num_correct": snum_correct,
                    "s_num_gold": snum_gold,
                    "el_micro_f1": inference_results.micro_entity_linking.f1.compute(),
                    "el_macro_f1": inference_results.macro_entity_linking.f1.compute(),
                    "md_micro_f1": inference_results.micro_mention_detection.f1.compute(),
                    "md_macro_f1": inference_results.macro_mention_detection.f1.compute(),
                    "epoch": epoch,
                    "allocated_memory": mm_alloc,
                })

    def get_mode_checkpoint_name(self):
        return os.path.join(self.checkpoints_root, "spel-step-3")


if __name__ == '__main__':
    try:
        b_annotator = FinetuneS3()
        b_annotator.finetune(checkpoint_name=None, n_epochs=60, batch_size=10, bert_dropout=0.2, label_size=10240,
                             eval_batch_size=2)
    finally:
        if TRACK_WITH_WANDB:
            wandb.finish()
