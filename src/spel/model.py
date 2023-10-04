"""
The implementation of the main annotator class from "SpEL: Structured Prediction for Entity Linking"
"""
import os
import re
import pickle
import numpy
from typing import List
from glob import glob
from itertools import chain

from transformers import AutoModelForMaskedLM
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from spel.utils import store_validation_data_wiki, chunk_annotate_and_merge_to_phrase, \
    get_aida_set_phrase_splitted_documents, compare_gold_and_predicted_annotation_documents
from spel.decao_eval import EntityEvaluationScores, InOutMentionEvaluationResult
from spel.span_annotation import SubwordAnnotation
from spel.data_loader import BERT_MODEL_NAME, dl_sa, tokenizer
from spel.configuration import get_checkpoints_dir, get_aida_train_canonical_redirects, get_ood_canonical_redirects, \
    get_logdir_dir, get_exec_run_file

class SpELAnnotator:
    def __init__(self):
        super(SpELAnnotator, self).__init__()
        self.checkpoints_root = get_checkpoints_dir()
        self.logdir = get_logdir_dir()
        self.exec_run_file = get_exec_run_file()

        self.text_chunk_length = 254
        self.text_chunk_overlap = 20

        self.bert_lm = None
        self.number_of_bert_layers = 0
        self.bert_lm_h = 0
        self.out = None
        self.softmax = None

    def init_model_from_scratch(self, base_model=BERT_MODEL_NAME, device="cpu"):
        """
        This is required to be called to load up the base model architecture before loading the fine-tuned checkpoint.
        """
        if base_model:
            self.bert_lm = AutoModelForMaskedLM.from_pretrained(base_model, output_hidden_states=True,
                                                                cache_dir=get_checkpoints_dir() / "hf").to(device)
            self.disable_roberta_lm_head()
            self.number_of_bert_layers = self.bert_lm.config.num_hidden_layers + 1
            self.bert_lm_h = self.bert_lm.config.hidden_size
            self.out = nn.Embedding(num_embeddings=len(dl_sa.mentions_vocab),
                                    embedding_dim=self.bert_lm_h, sparse=True).to(device)
            self.softmax = nn.Softmax(dim=-1)

    def shrink_classification_head_to_aida(self, device):
        """
        This will be called in fine-tuning step 3 to shrink the classification head to in-domain data vocabulary.
        """
        aida_mentions_vocab, aida_mentions_itos = dl_sa.get_aida_vocab_and_itos()
        if self.out_module.num_embeddings == len(aida_mentions_vocab):
            return
        current_state_dict = self.out_module.state_dict()
        new_out = nn.Embedding(num_embeddings=len(aida_mentions_vocab),
                               embedding_dim=self.bert_lm_h, sparse=True).to(device)
        new_state_dict = new_out.state_dict()
        for index_new in range(len(aida_mentions_itos)):
            item_new = aida_mentions_itos[index_new]
            assert item_new in dl_sa.mentions_vocab, \
                "the aida fine-tuned mention vocab must be a subset of the original vocab"
            index_current = dl_sa.mentions_vocab[item_new]
            new_state_dict['weight'][index_new] = current_state_dict['weight'][index_current]
        new_out.load_state_dict(new_state_dict, strict=False)
        self.out = new_out.to(device)
        dl_sa.shrink_vocab_to_aida()
        model_params = sum(p.numel() for p in self.bert_lm.parameters())
        out_params = sum(p.numel() for p in self.out.parameters())
        print(f' * Shrank model to {model_params+out_params} number of parameters ({model_params} parameters '
              f'for the encoder and {out_params} parameters for the classification head)!')

    @property
    def current_device(self):
        return self.lm_module.device

    @property
    def lm_module(self):
        return self.bert_lm.module if isinstance(self.bert_lm, nn.DataParallel) or \
                                      isinstance(self.bert_lm, nn.parallel.DistributedDataParallel) else self.bert_lm

    @property
    def out_module(self):
        return self.out.module if isinstance(self.out, nn.DataParallel) or \
                                  isinstance(self.out, nn.parallel.DistributedDataParallel) else self.out

    @staticmethod
    def get_canonical_redirects(limit_to_conll=True):
        return get_aida_train_canonical_redirects() if limit_to_conll else get_ood_canonical_redirects()

    def create_optimizers(self, encoder_lr=5e-5, decoder_lr=0.1, exclude_parameter_names_regex=None):
        if exclude_parameter_names_regex is not None:
            bert_lm_parameters = list()
            regex = re.compile(exclude_parameter_names_regex)
            for n, p in list(self.lm_module.named_parameters()):
                if not len(regex.findall(n)) > 0:
                    bert_lm_parameters.append(p)
        else:
            bert_lm_parameters = list(self.lm_module.parameters())
        bert_optim = optim.Adam(bert_lm_parameters, lr=encoder_lr)
        if decoder_lr < 1e-323:
            # IMPORTANT! This is a hack since if we don't consider an optimizer for the last layer(e.g. decoder_lr=0.0),
            #  BCEWithLogitsLoss will become unstable and memory will explode.
            decoder_lr = 1e-323
        out_optim = optim.SparseAdam(self.out.parameters(), lr=decoder_lr)
        return bert_optim, out_optim

    @staticmethod
    def create_warmup_scheduler(optimizer, warmup_steps):
        """
        Creates a scheduler which increases the :param optimizer: learning rate from 0 to the specified learning rate
            in :param warmup_steps: number of batches.
        You need to call scheduler.step() after optimizer.step() in your code for this scheduler to take effect
        """
        return optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: epoch / warmup_steps if epoch < warmup_steps else 1.0)

    def get_highest_confidence_model_predictions(self, batch_token_ids, topk_per_token=20, topk_from_batch=8196):
        """
        This function will be used for hard negative mining. For a given input batch, it will return
         the `topk_from_batch` mentions which have had model puzzled. In the process, to reduce the computational
          complexity the model will first select `topk_per_token` number of candidates from the vocabulary, and then
           applies the topk selection on it.
        """
        with torch.no_grad():
            logits = self.get_model_raw_logits_inference(batch_token_ids)
            # topk_logit_per_token, topk_eids_per_token = logits.topk(topk_per_token, sorted=False, dim=-1)
            # This is a workaround to the torch.topk bug for large sized tensors
            topk_logit_per_token, topk_eids_per_token = [], []
            for batch_item in logits:
                topk_probs, topk_ids = batch_item.topk(topk_per_token, sorted=False, dim=-1)
                topk_logit_per_token.append(topk_probs)
                topk_eids_per_token.append(topk_ids)
            topk_logit_per_token = torch.stack(topk_logit_per_token, dim=0)
            topk_eids_per_token = torch.stack(topk_eids_per_token, dim=0)
            i = torch.cat(
                [
                    topk_eids_per_token.view(1, -1),
                    torch.zeros(topk_eids_per_token.view(-1).size(), dtype=torch.long,
                                device=topk_eids_per_token.device).view(1, -1),
                ],
                dim=0,
            )
            v = topk_logit_per_token.view(-1)
            st = torch.sparse.FloatTensor(i, v)
            stc = st.coalesce()
            topk_indices = stc._values().sort(descending=True)[1][:topk_from_batch]
            result = stc._indices()[0, topk_indices]

            return result.cpu().tolist()
            # ###########################################################################################

    def annotate_subword_ids(self, subword_ids_list: List, k_for_top_k_to_keep: int, token_offsets=None) \
            -> List[SubwordAnnotation]:
        with torch.no_grad():
            token_ids = torch.LongTensor(subword_ids_list)
            raw_logits, hidden_states = self.get_model_raw_logits_inference(token_ids, return_hidden_states=True)
            logits = self.get_model_logits_inference(raw_logits, hidden_states, k_for_top_k_to_keep, token_offsets)
            return logits

    def get_model_raw_logits_training(self, token_ids, label_ids, label_probs):
        # label_probs is not used in this function but provided for the classes inheriting SpELAnnotator.
        enc = self.bert_lm(token_ids).hidden_states[-1]
        out = self.out(label_ids)
        logits = enc.matmul(out.transpose(0, 1))
        return logits

    def get_model_logits_inference(self, raw_logits, hidden_states, k_for_top_k_to_keep, token_offsets=None) \
            -> List[SubwordAnnotation]:
        # hidden_states is not used in this function but provided for the classes inheriting SpELAnnotator.
        logits = self.softmax(raw_logits)
        # The following line could possibly cause errors in torch version 1.13.1
        # see https://github.com/pytorch/pytorch/issues/95455 for more information
        top_k_logits, top_k_indices = logits.topk(k_for_top_k_to_keep)
        top_k_logits = top_k_logits.squeeze(0).cpu().tolist()
        top_k_indices = top_k_indices.squeeze(0).cpu().tolist()
        chunk = ["" for _ in top_k_logits] if token_offsets is None else token_offsets
        return [SubwordAnnotation(p, i, x[0]) for p, i, x in zip(top_k_logits, top_k_indices, chunk)]

    def get_model_raw_logits_inference(self, token_ids, return_hidden_states=False):
        encs = self.lm_module(token_ids.to(self.current_device)).hidden_states
        out = self.out_module.weight
        logits = encs[-1].matmul(out.transpose(0, 1))
        return (logits, encs) if return_hidden_states else logits

    def evaluate(self, epoch, batch_size, label_size, best_f1, is_training=True, use_retokenized_wikipedia_data=False,
                 potent_score_threshold=0.82):
        self.bert_lm.eval()
        self.out.eval()
        vocab_pad_id = dl_sa.mentions_vocab['<pad>']

        all_words, all_tags, all_y, all_y_hat, all_predicted, all_token_ids = [], [], [], [], [], []
        subword_eval = InOutMentionEvaluationResult(vocab_index_of_o=dl_sa.mentions_vocab['|||O|||'])
        dataset_name = store_validation_data_wiki(
            self.checkpoints_root, batch_size, label_size, is_training=is_training,
            use_retokenized_wikipedia_data=use_retokenized_wikipedia_data)
        with torch.no_grad():
            for d_file in tqdm(sorted(glob(os.path.join(self.checkpoints_root, dataset_name, "*")))):
                batch_token_ids, label_ids, label_probs, eval_mask, label_id_to_entity_id_dict, \
                    batch_entity_ids, is_in_mention, _ = pickle.load(open(d_file, "rb"))
                logits = self.get_model_raw_logits_inference(batch_token_ids)
                subword_eval.update_scores(eval_mask, is_in_mention, logits)
                y_hat = logits.argmax(-1)

                tags = list()
                predtags = list()
                y_resolved_list = list()
                y_hat_resolved_list = list()
                token_list = list()

                for batch_id, seq in enumerate(label_probs.max(-1)[1]):
                    for token_id, label_id in enumerate(seq[:-self.text_chunk_overlap]):
                        if eval_mask[batch_id][token_id].item() == 0:
                            y_resolved = vocab_pad_id
                        else:
                            y_resolved = label_ids[label_id].item()
                        y_resolved_list.append(y_resolved)
                        tags.append(dl_sa.mentions_itos[y_resolved])
                        y_hat_resolved = y_hat[batch_id][token_id].item()
                        y_hat_resolved_list.append(y_hat_resolved)
                        predtags.append(dl_sa.mentions_itos[y_hat_resolved])
                        token_list.append(batch_token_ids[batch_id][token_id].item())

                all_y.append(y_resolved_list)
                all_y_hat.append(y_hat_resolved_list)
                all_tags.append(tags)
                all_predicted.append(predtags)
                all_words.append(tokenizer.convert_ids_to_tokens(token_list))
                all_token_ids.append(token_list)
                del batch_token_ids, label_ids, label_probs, eval_mask, \
                    label_id_to_entity_id_dict, batch_entity_ids, logits, y_hat

        y_true = numpy.array(list(chain(*all_y)))
        y_pred = numpy.array(list(chain(*all_y_hat)))
        all_token_ids = numpy.array(list(chain(*all_token_ids)))

        num_proposed = len(y_pred[(1 < y_pred) & (all_token_ids > 0)])
        num_correct = (((y_true == y_pred) & (1 < y_true) & (all_token_ids > 0))).astype(int).sum()
        num_gold = len(y_true[(1 < y_true) & (all_token_ids > 0)])

        precision = num_correct / num_proposed if num_proposed > 0.0 else 0.0
        recall = num_correct / num_gold if num_gold > 0.0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
        f05 = 1.5 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
        if f1 > best_f1:
            print("Saving the best checkpoint ...")
            config = self.prepare_model_checkpoint(epoch)
            fname = self.get_mode_checkpoint_name()
            torch.save(config, f"{fname}.pt")
            print(f"weights were saved to {fname}.pt")
        if precision > potent_score_threshold and recall > potent_score_threshold and is_training:
            print(f"Saving the potent checkpoint with both precision and recall above {potent_score_threshold} ...")
            config = self.prepare_model_checkpoint(epoch)
            try:
                fname = self.get_mode_checkpoint_name()
                torch.save(config, f"{fname}-potent.pt")
                print(f"weights were saved to {fname}-potent.pt")
            except NotImplementedError:
                pass
        self.bert_lm.train()
        self.out.train()
        with open(self.exec_run_file, "a+") as exec_file:
            exec_file.write(f"{precision}, {recall}, {f1}, {f05}, {num_proposed}, {num_correct}, {num_gold}, "
                            f"{epoch+1},,\n")
        return precision, recall, f1, f05, num_proposed, num_correct, num_gold, subword_eval

    def inference_evaluate(self, epoch, best_f1, dataset_name='testa'):
        self.bert_lm.eval()
        self.out.eval()
        evaluation_results = EntityEvaluationScores(dataset_name)
        gold_documents = get_aida_set_phrase_splitted_documents(dataset_name)
        for gold_document in tqdm(gold_documents):
            t_sentence = " ".join([x.word_string for x in gold_document])
            predicted_document = chunk_annotate_and_merge_to_phrase(self, t_sentence, k_for_top_k_to_keep=1)
            comparison_results = compare_gold_and_predicted_annotation_documents(gold_document, predicted_document)
            g_md = set((e[1].begin_character, e[1].end_character)
                       for e in comparison_results if e[0].resolved_annotation)
            p_md = set((e[1].begin_character, e[1].end_character)
                       for e in comparison_results if e[1].resolved_annotation)
            g_el = set((e[1].begin_character, e[1].end_character, dl_sa.mentions_itos[e[0].resolved_annotation])
                       for e in comparison_results if e[0].resolved_annotation)
            p_el = set((e[1].begin_character, e[1].end_character, dl_sa.mentions_itos[e[1].resolved_annotation])
                       for e in comparison_results if e[1].resolved_annotation)
            if p_el:
                evaluation_results.record_mention_detection_results(p_md, g_md)
                evaluation_results.record_entity_linking_results(p_el, g_el)
        if evaluation_results.micro_entity_linking.f1.compute() > best_f1:
            print("Saving the best checkpoint ...")
            config = self.prepare_model_checkpoint(epoch)
            fname = self.get_mode_checkpoint_name()
            torch.save(config, f"{fname}.pt")
            print(f"weights were saved to {fname}.pt")
        self.bert_lm.train()
        self.out.train()
        return evaluation_results

    def prepare_model_checkpoint(self, epoch):
        chk_point = {
            "bert_lm": self.lm_module.state_dict(),
            "number_of_bert_layers": self.number_of_bert_layers,
            "bert_lm_h": self.bert_lm_h,
            "out": self.out_module.state_dict(),
            "epoch": epoch,
        }
        sub_model_specific_checkpoint_data = self.sub_model_specific_checkpoint_data()
        for key in sub_model_specific_checkpoint_data:
            assert key not in ["bert_lm", "number_of_bert_layers", "bert_lm_h", "out", "epoch"], \
                f"{key} is already considered in prepare_model_checkpoint function"
            chk_point[key] = sub_model_specific_checkpoint_data[key]
        return chk_point

    def disable_roberta_lm_head(self):
        assert self.bert_lm is not None
        self.bert_lm.lm_head.layer_norm.bias.requires_grad = False
        self.bert_lm.lm_head.layer_norm.weight.requires_grad = False
        self.bert_lm.lm_head.dense.bias.requires_grad = False
        self.bert_lm.lm_head.dense.weight.requires_grad = False
        self.bert_lm.lm_head.decoder.bias.requires_grad = False

    def _load_from_checkpoint_object(self, checkpoint, device="cpu"):
        torch.cuda.empty_cache()
        self.bert_lm.load_state_dict(checkpoint["bert_lm"], strict=False)
        self.bert_lm.to(device)
        self.disable_roberta_lm_head()
        self.out.load_state_dict(checkpoint["out"], strict=False)
        self.out.to(device)
        self.number_of_bert_layers = checkpoint["number_of_bert_layers"]
        self.bert_lm_h = checkpoint["bert_lm_h"]
        self.sub_model_specific_load_checkpoint_data(checkpoint)
        self.bert_lm.eval()
        self.out.eval()
        model_params = sum(p.numel() for p in self.bert_lm.parameters())
        out_params = sum(p.numel() for p in self.out.parameters())
        print(f' * Loaded model with {model_params+out_params} number of parameters ({model_params} parameters '
              f'for the encoder and {out_params} parameters for the classification head)!')

    @staticmethod
    def download_from_torch_hub(finetuned_after_step=1):
        assert 4 >= finetuned_after_step >= 1
        if finetuned_after_step == 4:
            # This model is the same SpEL finetuned model after step 3 except that its classification layer projects to
            # the entirety of the step-2 model rather than shrinking it in size
            file_name = "spel-base-step-3-500K.pt"
            # Downloads and returns the finetuned model checkpoint created on Oct-03-2023
            checkpoint = torch.hub.load_state_dict_from_url('https://vault.sfu.ca/index.php/s/8nw5fFXdz2yBP5z/download',
                                                            model_dir=str(get_checkpoints_dir()), map_location="cpu",
                                                            file_name=file_name)
        elif finetuned_after_step == 3:
            file_name = "spel-base-step-3.pt"
            # Downloads and returns the finetuned model checkpoint created on Sep-26-2023 with P=92.06|R=91.93|F1=91.99
            checkpoint = torch.hub.load_state_dict_from_url('https://vault.sfu.ca/index.php/s/HpQ3PMm6A3y1NBl/download',
                                                            model_dir=str(get_checkpoints_dir()), map_location="cpu",
                                                            file_name=file_name)
        elif finetuned_after_step == 2:
            file_name = 'spel-base-step-2.pt'
            # Downloads and returns the pretrained model checkpoint created on Sep-26-2023 with P=77.60|R=77.91|F1=77.75
            checkpoint = torch.hub.load_state_dict_from_url('https://vault.sfu.ca/index.php/s/Hf37vc1foluHPBh/download',
                                                            model_dir=str(get_checkpoints_dir()), map_location="cpu",
                                                            file_name=file_name)
        else:
            file_name = 'spel-base-step-1.pt'
            # Downloads and returns the pretrained model checkpoint created on Sep-11-2023 with P=82.50|R=83.16|F1=82.83
            checkpoint = torch.hub.load_state_dict_from_url('https://vault.sfu.ca/index.php/s/9OAoAG5eYeREE9V/download',
                                                            model_dir=str(get_checkpoints_dir()), map_location="cpu",
                                                            file_name=file_name)
        print(f" * Loaded pretrained model checkpoint: {file_name}")
        return checkpoint

    @staticmethod
    def download_large_from_torch_hub(finetuned_after_step=1):
        assert 4 >= finetuned_after_step >= 1
        if finetuned_after_step == 4:
            # This model is the same SpEL finetuned model after step 3 except that its classification layer projects to
            # the entirety of the step-2 model rather than shrinking it in size
            file_name = "spel-large-step-3-500K.pt"
            # Downloads and returns the finetuned model checkpoint created on Oct-03-2023
            checkpoint = torch.hub.load_state_dict_from_url('https://vault.sfu.ca/index.php/s/BCvputD1ByAvILC/download',
                                                            model_dir=str(get_checkpoints_dir()), map_location="cpu",
                                                            file_name=file_name)
        elif finetuned_after_step == 3:
            file_name = "spel-large-step-3.pt"
            # Downloads and returns the finetuned model checkpoint created on Oct-02-2023 with P=92.53|R=92.99|F1=93.76
            checkpoint = torch.hub.load_state_dict_from_url('https://vault.sfu.ca/index.php/s/kBBlYVM4Tr59P0q/download',
                                                            model_dir=str(get_checkpoints_dir()), map_location="cpu",
                                                            file_name=file_name)
        elif finetuned_after_step == 2:
            file_name = 'spel-large-step-2.pt'
            # Downloads and returns the pretrained model checkpoint created on Oct-02-2023 with P=77.36|R=73.11|F1=75.18
            checkpoint = torch.hub.load_state_dict_from_url('https://vault.sfu.ca/index.php/s/rnDiuKns7gzADyb/download',
                                                            model_dir=str(get_checkpoints_dir()), map_location="cpu",
                                                            file_name=file_name)
        else:
            file_name = 'spel-large-step-1.pt'
            # Downloads and returns the pretrained model checkpoint created on Sep-11-2023 with P=84.02|R=82.74|F1=83.37
            checkpoint = torch.hub.load_state_dict_from_url('https://vault.sfu.ca/index.php/s/bTp6UN2xL7Yh52w/download',
                                                            model_dir=str(get_checkpoints_dir()), map_location="cpu",
                                                            file_name=file_name)
        print(f" * Loaded pretrained model checkpoint: {file_name}")
        return checkpoint


    def load_checkpoint(self, checkpoint_name, device="cpu", rank=0, load_from_torch_hub=False, finetuned_after_step=1):
        if load_from_torch_hub and BERT_MODEL_NAME == "roberta-large":
            checkpoint = self.download_large_from_torch_hub(finetuned_after_step)
            self._load_from_checkpoint_object(checkpoint, device)
        elif load_from_torch_hub and BERT_MODEL_NAME == "roberta-base":
            checkpoint = self.download_from_torch_hub(finetuned_after_step)
            self._load_from_checkpoint_object(checkpoint, device)
        else: # load from the local .checkpoints directory
            if rank == 0:
                print("Loading model checkpoint: {}".format(checkpoint_name))
            fname = os.path.join(self.checkpoints_root, checkpoint_name)
            checkpoint = torch.load(fname, map_location="cpu")
            self._load_from_checkpoint_object(checkpoint, device)

    # #############################FUNCTIONS THAT THE SUB-MODELS MUST REIMPLEMENT####################################
    def sub_model_specific_checkpoint_data(self):
        """
        :return: a dictionary of key values containing everything that matters to the sub-model and is not already
            considered in prepare_model_checkpoint.
        """
        return {}

    def sub_model_specific_load_checkpoint_data(self, checkpoint):
        return

    def get_mode_checkpoint_name(self):
        raise NotImplementedError

    def annotate(self, nif_collection, **kwargs):
        raise NotImplementedError
