"""
Data structure classes required and used for multiple levels of granularity in spans.
"""
from spel.data_loader import dl_sa
from mosestokenizer import MosesDetokenizer
detokenize = MosesDetokenizer('en')


class PhraseAnnotation:
    def __init__(self, initial_word):
        self.words = [initial_word]
        self._resolved_annotation = initial_word.resolved_annotation
        self.ppr_for_ned_candidates = initial_word.ppr_for_ned_candidates

    @property
    def has_valid_bioes_labels(self):
        # B = 0, I = 1, O = 2, E = 3, S = 4
        return all([x.has_valid_bioes_labels and x.bioes_labels is not None for x in self.words])

    def add(self, word):
        self.words.append(word)
        # There are some phrases that are annotated as O but have PPRforNED candidates, those will be ignored!
        if self._resolved_annotation > 0 and self.ppr_for_ned_candidates != word.ppr_for_ned_candidates:
            self.ppr_for_ned_candidates = list(set(self.ppr_for_ned_candidates) & set(word.ppr_for_ned_candidates))

    def all_possible_annotations(self):
        all_common_ids = set.intersection(*[set([y[0] for y in x.candidates]) for x in self.words])
        all_common_ids_average_confidence = map(lambda x: sum(x)/len(x), [
            [sum(y[1])/len(y[1]) for x in self.words for y in x.candidates if y[0] == k] for k in all_common_ids])
        return sorted(zip(all_common_ids, all_common_ids_average_confidence), key=lambda x: x[1], reverse=True)

    def set_alternative_as_resolved_annotation(self, alternative):
        self._resolved_annotation = alternative

    @property
    def resolved_annotation(self):
        return self._resolved_annotation

    @property
    def subword_annotations(self):
        return [x for w in self.words for x in w.annotations]

    @property
    def word_string(self):
        return detokenize([x.word_string.replace("\n", "\u010a").replace("Â£", "£").replace("âĦ¢", '™')
                          .replace('Ã¼','ü').replace('Ã©', 'é').replace('ÃŃ', 'í') for x in self.words])

    @property
    def begin_character(self):
        return self.words[0].token_offsets[0][1][0]

    @property
    def end_character(self):
        return self.words[-1].token_offsets[-1][1][-1]

    @property
    def average_annotation_confidence(self):
        ac = [x.resolved_annotation_confidence for x in self.words]
        return sum(ac) / len(ac)

    def __str__(self):
        return f"{self.word_string} ({self.begin_character}, {self.end_character}) | annotation: " \
               f"{self.words[0].annotations[0].idx2tag[self.resolved_annotation]}"


class WordAnnotation:
    def __init__(self, subword_annotations, token_offsets, ppr_for_ned_candidates=None):
        if ppr_for_ned_candidates is None:
            ppr_for_ned_candidates = []
        self.annotations = subword_annotations
        self.token_offsets = token_offsets
        self.ppr_for_ned_candidates = ppr_for_ned_candidates
        self.is_valid_annotation = False if not subword_annotations else True
        self.word_string = ''.join([x[0].replace('\u0120', '') for x in token_offsets])
        # even if self.is_valid_annotation is True we could still have the candidates to be empty
        #   since there could be no consensus among the subword predictions.
        self.candidates = sorted([] if not self.is_valid_annotation else [
            (cid, self._get_assigned_probabilities(cid)) for cid in set.intersection(*[set(y.top_k_i_list)
                                                                                       for y in self.annotations])],
                                 key=lambda x: sum(x[1])/len(x[1]), reverse=True)
        self.resolved_annotation = self._resolve_annotation()
        rc = self._get_assigned_probabilities(self.resolved_annotation)
        self.resolved_annotation_confidence = sum(rc) / len(rc)
        if not self.candidates:
            self.candidates = [(self.resolved_annotation, rc)]
        assert self.resolved_annotation in [x[0] for x in self.candidates]
        self.has_valid_bioes_labels = all([x.has_valid_bioes_label for x in self.annotations])
        self.bioes_labels = None if not self.has_valid_bioes_labels else [x.bioes_label for x in self.annotations]

    def _resolve_annotation(self):
        if not self.is_valid_annotation:
            return 0
        r = [x.item() for x in self.annotations]
        if r.count(r[0]) == len(r):
            annotation = r[0]
        elif self.candidates:
            # here we return the annotation with the highest average probability prediction over all the subwords
            annotation = self.candidates[0][0]
        else:
            # here we return the annotation which the model has predicted as highest probability for
            #   the majority of the subwords
            most_frequent = max(set(r), key=r.count)
            if r.count(most_frequent) == 1:
                annotation = r[0]
            else:
                annotation = most_frequent
        return annotation

    def _get_assigned_probabilities(self, cid):
        assigned_probabilities = []
        for a in self.annotations:
            found = False
            for i, p in zip(a.top_k_i_list, a.top_k_p_list):
                if i == cid:
                    assigned_probabilities.append(p)
                    found = True
                    break
            if not found:
                assigned_probabilities.append(0.0)
        assert len(assigned_probabilities) == len(self.annotations)
        return assigned_probabilities

    def __str__(self):
        ann = self.annotations[0].idx2tag[self.resolved_annotation]
        cdns = ','.join([f'({self.annotations[0].idx2tag[x[0]]}: {sum(x[1])/len(x[1])})' for x in self.candidates])
        return f"{self.word_string} | annotation: {ann} | candidates: [{cdns}]"


class SubwordAnnotation:
    """
    The value of his class will be equal to the value of its "self.top_k_i_list[0]", the rest of the information will be
     carried over for future decision-making and evaluation.
    """
    def __init__(self, top_k_p_list, top_k_i_list, subword_string):
        self.top_k_p_list = top_k_p_list
        self.top_k_i_list = top_k_i_list
        subword_string = "UNDEF_STR" if not subword_string else subword_string
        self.subword_string = subword_string.replace('\u0120', '')
        self.bioes_label = 2
        self.has_valid_bioes_label = False
        self.bioes_probabilities = None

    def __eq__(self, other):
        if isinstance(other, int):
            return self.top_k_i_list[0] == other
        elif isinstance(other, SubwordAnnotation):
            return self.top_k_i_list[0] == other.top_k_i_list[0]
        else:
            raise ValueError

    def __str__(self):
        return f"({self.subword_string}, <<" \
               f"{'>> <<'.join([f'{dl_sa.mentions_itos[i]}: {p:.3f}' for i, p in zip(self.top_k_i_list, self.top_k_p_list)])}>>)"

    def item(self):
        return self.top_k_i_list[0]

    def item_probability(self):
        return self.top_k_p_list[0]

    def set_bioes_label(self, label: int, probs: list):
        assert 0 <= label <= 5
        assert len(probs) == 5
        self.has_valid_bioes_label = True
        self.bioes_label = label
        self.bioes_probabilities = probs
