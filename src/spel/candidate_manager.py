"""
This file contains the implementation of the candidate manager in charge of loading the candidate sets,
 and modifying the phrase annotations using the loaded candidates.
"""
import json
from spel.span_annotation import PhraseAnnotation
from spel.configuration import get_resources_dir


class CandidateManager:
    def __init__(self, mentions_vocab, is_kb_yago = False, is_ppr_for_ned = False, is_context_agnostic = False,
                 is_indexed_for_spans= False):
        self.mentions_vocab = mentions_vocab
        self.candidates = None
        if is_kb_yago:
            print(" * Loading the candidates stored for KB+YAGO ...")
            is_context_agnostic = True
            is_indexed_for_spans = False
            self.load_kb_plus_yago()
        elif is_ppr_for_ned:
            print(" * Loading the {} PPRforNED candidate set ...".format(
                'context agnostic' if is_context_agnostic else 'context aware'))
            self.load_ppr_for_ned_candidates(is_context_agnostic, is_indexed_for_spans)
        else:
            raise ValueError("Either \'is_kb_yago\' or \'is_ppr_for_ned\' flags must be True!")
        self.is_context_agnostic = is_context_agnostic
        self.is_indexed_for_spans = is_indexed_for_spans
        self.is_kb_yago = is_kb_yago
        self.is_ppr_for_ned = is_ppr_for_ned

    def load_ppr_for_ned_candidates(self, is_context_agnostic, is_indexed_for_spans):
        if is_context_agnostic:
            file_address = "context_agnostic_mentions.json"
        elif is_indexed_for_spans:
            file_address = "context_aware_spans.json"
        else:
            file_address = "context_aware_mentions.json"
        candidates_a = json.load(open(
            get_resources_dir() / "data" / "candidates" / "aida_testa_pprforned" / file_address, "r"))
        candidates_b = json.load(open(
            get_resources_dir() / "data" / "candidates" / "aida_testb_pprforned" / file_address, "r"))
        if is_context_agnostic:
            for key in candidates_b:
                if key in candidates_a:
                    for elem in candidates_b[key]:
                        if elem not in candidates_a[key]:
                            candidates_a[key].append(elem)
                else:
                    candidates_a[key] = candidates_b[key]
        else:
            candidates_a.update(candidates_b)
        self.candidates = candidates_a

    def load_kb_plus_yago(self):
        self.candidates = json.load(open(
            get_resources_dir() / "data" / "candidates" / "kb_plus_yago_candidates.json", "r"))

    def _fetch_candidates(self, phrase_annotation, sentence = None):
        candidates = []
        if self.is_kb_yago:
            phrase_to_check = phrase_annotation.word_string.lower()
            if phrase_to_check in self.candidates:
                candidates = self.candidates[phrase_to_check]
        elif self.is_ppr_for_ned:
            # TODO lower-cased check mention surface forms
            span_key  = f"({phrase_annotation.begin_character}, {phrase_annotation.end_character})"
            if self.is_context_agnostic and phrase_annotation.word_string in self.candidates:
                candidates = self.candidates[phrase_annotation.word_string]
            elif not self.is_context_agnostic and sentence in self.candidates:
                if self.is_indexed_for_spans and span_key in self.candidates[sentence]:
                    candidates = self.candidates[sentence][span_key]
                elif not self.is_indexed_for_spans and phrase_annotation.word_string in self.candidates[sentence]:
                    candidates = self.candidates[sentence][phrase_annotation.word_string]
        return candidates

    def modify_phrase_annotation_using_candidates(self, phrase_annotation: PhraseAnnotation, sentence: str = None):
        """
        The method post processes the :param phrase_annotation: found in a :param sentence: to make sure it is bound to
          the predefined {self.candidates} set.
        It is not possible to perform candidate look up for spans in context agnostic scenario
          so {self.is_indexed_for_spans} will only be considered where {self.is_context_agnostic} is False.
        """
        if self.candidates is None or phrase_annotation.resolved_annotation == 0:
            return
        candidates = self._fetch_candidates(phrase_annotation, sentence)
        if not candidates:
            phrase_annotation.set_alternative_as_resolved_annotation(0)
            return
        if self.is_kb_yago:
            candidates_ = [self.mentions_vocab[x[0]] for x in candidates if x[0] in self.mentions_vocab]
            prior_probabilities_ = [x[1] for x in candidates if x[0] in self.mentions_vocab]
        else:
            candidates_ = [self.mentions_vocab[x] for x in candidates if x in self.mentions_vocab]
            prior_probabilities_ = [1.0 for x in candidates if x in self.mentions_vocab]
        # TODO use the prior_probabilities_ to adjust the probabilities
        if candidates_:
            all_p_anns = phrase_annotation.all_possible_annotations()
            filtered_p_predictions = sorted(
                [x for x in all_p_anns if x[0] in candidates_], key=lambda y: y[1], reverse=True)
            if filtered_p_predictions:
                phrase_annotation.set_alternative_as_resolved_annotation(filtered_p_predictions[0][0])
            else:
                phrase_annotation.set_alternative_as_resolved_annotation(0)