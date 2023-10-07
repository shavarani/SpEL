"""
This script uses the EntityEvaluationScores which is a minor add-on to the evaluation script released by Nicola De Cao:
    https://github.com/nicola-decao/efficient-autoregressive-EL
and performs a local evaluation of the SpEL fine-tuned models (in different fine-tuning steps).
Please note that the numbers that this file spits out are not comparable to our reported numbers in the
    "SpEL: Structured Prediction for Entity Linking" since we chunk gold spans and align the predictions and actuals.

However, SpELEvaluator.annotate implements the correct interface which can be used by the server integrating
 `gerbil_coonect` code to communicate with GERBIL.
"""
from tqdm import tqdm

from spel.model import SpELAnnotator
from spel.utils import chunk_annotate_and_merge_to_phrase, postprocess_annotations, \
    get_aida_set_phrase_splitted_documents, compare_gold_and_predicted_annotation_documents
from spel.decao_eval import EntityEvaluationScores
from spel.data_loader import dl_sa
from spel.configuration import device

FINAL_ANNOTATION_POSTPROCESSING_ALLOWED = True
BERT_ANNOTATOR_CHECKPOINT = None


class SpELEvaluator(SpELAnnotator):
    def __init__(self):
        super(SpELEvaluator, self).__init__()

    def annotate(self, nif_collection, **kwargs):
        """The function which is called when a GERBIL document is passed to el.spel annotator"""
        assert len(nif_collection.contexts) == 1
        context = nif_collection.contexts[0]
        ignore_non_aida_vocab = kwargs["ignore_non_aida_vocab"] if "ignore_non_aida_vocab" in kwargs else True
        # kb_prefix can either be a single string applied to any prediction or a dictionary defining one specific prefix
        #  for each possible annotation. This dict object MUST contain one entry with the key='[defalt_prefix]' which
        #   will be used for the cases that a model prediction is not found in the object.
        kb_prefix = kwargs["kb_prefix"] if "kb_prefix" in kwargs else 'http://en.wikipedia.org/wiki/'
        candidates_manager = kwargs["candidates_manager"] \
            if "candidates_manager" in kwargs and kwargs["candidates_manager"] else None
        phrase_annotations = chunk_annotate_and_merge_to_phrase(
            self, context.mention, k_for_top_k_to_keep=1,
            normalize_for_chinese_characters=True)
        if candidates_manager:
            [candidates_manager.modify_phrase_annotation_using_candidates(p, context.mention)
             for p in phrase_annotations]
        last_step_annotations = [[p.words[0].token_offsets[0][1][0],
                                  p.words[-1].token_offsets[-1][1][-1],
                                  (dl_sa.mentions_itos[p.resolved_annotation], p.subword_annotations)]
                                 for p in phrase_annotations if p.resolved_annotation != 0]
        if FINAL_ANNOTATION_POSTPROCESSING_ALLOWED:
            last_step_annotations = postprocess_annotations(last_step_annotations, context.mention)

        canonical_redirects = self.get_canonical_redirects(ignore_non_aida_vocab)

        for l_ann in [(l_ann[0], l_ann[1], (
                canonical_redirects[l_ann[2][0]], l_ann[2][1]) if l_ann[2][0] in canonical_redirects else l_ann[2])
                      for l_ann in last_step_annotations]:
            try:
                kbp = kb_prefix[l_ann[2][0]] if type(kb_prefix) == dict else kb_prefix
            except KeyError:
                kbp = kb_prefix['[defalt_prefix]']
            context.add_phrase(
                beginIndex=l_ann[0],
                endIndex=l_ann[1],
                score=sum([x.item_probability() for x in l_ann[2][1]])/len(l_ann[2][1]),
                annotator='http://sfu.ca/spel/annotator',
                taIdentRef=kbp+l_ann[2][0].replace("\"", "%22"))

    def get_model_raw_logits_inference(self, token_ids, return_hidden_states=False):
        encs = self.lm_module(token_ids.to(self.current_device)).hidden_states
        out = self.out_module.weight
        logits = encs[-1].matmul(out.transpose(0, 1))
        # The following line can provide the functionality to mask out any subset of undesired output entities from
        #  the model predictions on each subword in inference time. You don't need to uncomment it if you are just
        # interested in testing SpEL out.
        # logits = dl_sa.get_all_vocab_mask_for_aida().unsqueeze(0).unsqueeze(0).repeat(
        #       logits.size(0), logits.size(1), 1) + logits
        return (logits, encs) if return_hidden_states else logits

    def aida_conll_evaluate(self, checkpoint_name, k_for_top_k_to_keep=5, ignore_over_generated=False,
                            ignore_predictions_outside_candidate_list=False):
        self.init_model_from_scratch(device=device)
        self.shrink_classification_head_to_aida(device=device)
        if checkpoint_name is None:
            print('Loading the model which is fine-tuned on AIDA/CoNLL dataset ...')
            self.load_checkpoint(None, device=device, load_from_torch_hub=True, finetuned_after_step=3)
        else:
            self.load_checkpoint(checkpoint_name, device=device)
        for dataset_name in ['testa', 'testb']:
            evaluation_results = EntityEvaluationScores(dataset_name)
            gold_documents = get_aida_set_phrase_splitted_documents(dataset_name)
            for gold_document in tqdm(gold_documents):
                t_sentence = " ".join([x.word_string for x in gold_document])
                predicted_document = chunk_annotate_and_merge_to_phrase(
                    self, t_sentence, k_for_top_k_to_keep=k_for_top_k_to_keep)
                comparison_results = compare_gold_and_predicted_annotation_documents(
                    gold_document, predicted_document, ignore_over_generated=ignore_over_generated,
                    ignore_predictions_outside_candidate_list=ignore_predictions_outside_candidate_list)
                g_ed = set((e[1].begin_character, e[1].end_character)
                           for e in comparison_results if e[0].resolved_annotation)
                p_ed = set((e[1].begin_character, e[1].end_character)
                           for e in comparison_results if e[1].resolved_annotation)
                g_el = set((e[1].begin_character, e[1].end_character, dl_sa.mentions_itos[e[0].resolved_annotation])
                           for e in comparison_results if e[0].resolved_annotation)
                p_el = set((e[1].begin_character, e[1].end_character, dl_sa.mentions_itos[e[1].resolved_annotation])
                           for e in comparison_results if e[1].resolved_annotation)
                if p_el:
                    evaluation_results.record_mention_detection_results(p_ed, g_ed)
                    evaluation_results.record_entity_linking_results(p_el, g_el)
            print(evaluation_results)

    def wikipedia_evaluate(self, checkpoint_name, use_retokenized_wikipedia_data=False):
        self.init_model_from_scratch(device=device)
        if checkpoint_name is None:
            print(f'Loading the model which is not fine-tuned on AIDA/CoNLL dataset '
                  f'(retokenized={use_retokenized_wikipedia_data}) ...')
            self.load_checkpoint(None, device=device, load_from_torch_hub=True,
                                 finetuned_after_step=2 if use_retokenized_wikipedia_data else 1)
        else:
            self.load_checkpoint(checkpoint_name, device=device)
        # the next line will create SpEL/.checkpoints/validation_data_cache_b_2_l_1024_rt_wiki directory:
        precision, recall, f1, f05, num_proposed, num_correct, num_gold, subword_eval = self.evaluate(
            0, 2, 1024, 1.1, True, use_retokenized_wikipedia_data)
        print(f"Subword-level evaluation results on wikipedia validation set: "
              f"precision={precision:.5f}, recall={recall:.5f}, f1={f1:.5f}, f05={f05:.5f}, "
              f"num_proposed={num_proposed}, num_correct={num_correct}, num_gold={num_gold}")
        print(subword_eval)

    def fine_tuned_evaluate(self, checkpoint_name):
        self.init_model_from_scratch(device=device)
        self.shrink_classification_head_to_aida(device=device)
        if checkpoint_name is None:
            print('Loading the model which is fine-tuned on AIDA/CoNLL dataset ...')
            self.load_checkpoint(None, device=device, load_from_torch_hub=True, finetuned_after_step=3)
        else:
            self.load_checkpoint(checkpoint_name, device=device)
        precision, recall, f1, f05, num_proposed, num_correct, num_gold, subword_eval = self.evaluate(0, 2, 1024, 1.1, False)
        print(f"Subword-level evaluation results on wikipedia validation set: "
              f"precision={precision:.5f}, recall={recall:.5f}, f1={f1:.5f}, f05={f05:.5f}, "
              f"num_proposed={num_proposed}, num_correct={num_correct}, num_gold={num_gold}")
        print(subword_eval)


if __name__ == '__main__':
    b_annotator = SpELEvaluator()
    b_annotator.wikipedia_evaluate(checkpoint_name=BERT_ANNOTATOR_CHECKPOINT, use_retokenized_wikipedia_data=True)
    b_annotator.fine_tuned_evaluate(checkpoint_name=BERT_ANNOTATOR_CHECKPOINT)
    b_annotator.aida_conll_evaluate(checkpoint_name=BERT_ANNOTATOR_CHECKPOINT)
