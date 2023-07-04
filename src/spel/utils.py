"""
This file provides many functionalities that can be shared among different components.
The most important function in this file is `chunk_annotate_and_merge_to_phrase` which recieves a model and a raw text,
 annotates the text, and returns the annotation spans.
"""
import os
import json
import pickle
import string
from enum import Enum
from tqdm import tqdm

from spel.data_loader import get_dataset, tokenizer, dl_sa
from spel.span_annotation import SubwordAnnotation, WordAnnotation, PhraseAnnotation
from spel.aida import AIDADataset
from spel.configuration import get_resources_dir
from mosestokenizer import MosesTokenizer, MosesPunctuationNormalizer
moses_tokenize = MosesTokenizer('en', old_version=True)
normalize = MosesPunctuationNormalizer('en')


def get_punc_tokenized_words_list(word_list: list, labels_list: list = None):
    tokens = []
    labels = []
    for w_ind, o_token in enumerate(word_list):
        if o_token[0] not in string.punctuation and o_token[-1] not in string.punctuation:
            tokens.append(o_token)
            if labels_list:
                labels.append(labels_list[w_ind])
            if o_token.endswith("\'s") or o_token.endswith("\'S"):
                tokens[-1] = tokens[-1][:-2]
                tokens.append(o_token[-2:])
                if labels_list:
                    labels.append(labels_list[w_ind])
            continue
        # cases where the tokens start or end with punctuation
        before_tokens = []
        after_tokens = []
        while o_token and o_token[0] in string.punctuation:
            before_tokens.append(o_token[0])
            o_token = o_token[1:]
        while o_token and o_token[-1] in string.punctuation:
            after_tokens.append(o_token[-1])
            o_token = o_token[:-1]
        if before_tokens:
            tokens.append("".join(before_tokens))
            if labels_list:
                labels.append(labels_list[w_ind])
        if o_token:
            tokens.append(o_token)
            if labels_list:
                labels.append(labels_list[w_ind])
        if after_tokens:
            tokens.append("".join(after_tokens[::-1]))
            if labels_list:
                labels.append(labels_list[w_ind])
    if labels_list:
        return tokens, labels
    return tokens


def save_predictions_result(logdir, epoch, precision, recall, f1, num_proposed, num_correct, num_gold,
                            all_words, all_tags, all_y_hat, all_predicted):
    final = logdir + "/%s.P%.2f_R%.2f_F%.2f" % ("{}".format(str(epoch)), precision, recall, f1,)
    with open(final, "w") as fout:
        for words, tags, y_hat, preds in zip(all_words, all_tags, all_y_hat, all_predicted):
            assert len(preds) == len(words) == len(tags)
            for w, t, p in zip(words, tags, preds):
                if w == '<s>' or t == '<pad>':
                    continue
                fout.write(f"{w}\t{t}\t{p}\n")
            fout.write("\n")
        fout.write(f"num_proposed={num_proposed}\n")
        fout.write(f"num_correct={num_correct}\n")
        fout.write(f"num_gold={num_gold}\n")
        fout.write(f"precision={precision}\n")
        fout.write(f"recall={recall}\n")
        fout.write(f"f1={f1}\n")


def get_subword_to_word_mapping(subword_tokens, original_string, sequence_starts_and_ends_with_bos_eos=True):
    # subword_tokens starts with <s> and ends with </s>
    if sequence_starts_and_ends_with_bos_eos:
        subword_tokens = subword_tokens[1:-1]
    subword_to_word_mapping = []
    start_subword_index = 0
    end_subword_index = 0
    original_tokens = get_punc_tokenized_words_list(original_string.split())

    original_pointer = 0
    while len(subword_to_word_mapping) != len(original_tokens):
        next_t = tokenizer.convert_tokens_to_string(subword_tokens[start_subword_index:end_subword_index])
        next_t = next_t.strip()
        if next_t == original_tokens[original_pointer]:
            subword_to_word_mapping.append((start_subword_index, end_subword_index))
            original_pointer += 1
            start_subword_index = end_subword_index
        else:
            end_subword_index += 1
        if end_subword_index - start_subword_index > 1000:
            for i in [0, 1, 2, 3, 4]:
                n = tokenizer.convert_tokens_to_string(subword_tokens[start_subword_index:start_subword_index + 2 + i]).strip()
                o = "".join(original_tokens[original_pointer: original_pointer + 2]).replace("`", "\'")
                if n == o or n.replace(" ", "") == o.replace(" ", ""):
                    subword_to_word_mapping.append((start_subword_index, start_subword_index + 1))
                    original_pointer += 1
                    start_subword_index = start_subword_index + 1
                    subword_to_word_mapping.append((start_subword_index, start_subword_index + 1 + i))
                    original_pointer += 1
                    start_subword_index = start_subword_index + 1 + i
                    end_subword_index = start_subword_index
                    break
    return subword_to_word_mapping


def store_validation_data_wiki(checkpoints_root, batch_size, label_size, is_training, use_retokenized_wikipedia_data):
    dataset_name = f"validation_data_cache_b_{batch_size}_l_{label_size}_" \
                   f"{('rt_wiki' if use_retokenized_wikipedia_data else 'wiki') if is_training else 'conll'}/"
    if not os.path.exists(os.path.join(checkpoints_root, dataset_name)):
        os.mkdir(os.path.join(checkpoints_root, dataset_name))
    else:
        print("Retrieving the validation data ...")
        return dataset_name
    print("Caching the validation data ...")
    if is_training:
        valid_iter = tqdm(get_dataset(
            dataset_name='enwiki', split='valid', batch_size=batch_size, label_size=label_size,
            use_retokenized_wikipedia_data=use_retokenized_wikipedia_data))
    else:
        valid_iter = tqdm(get_dataset(dataset_name='aida', split='valid', batch_size=batch_size, label_size=label_size))
    for ind, (inputs, subword_mentions) in enumerate(valid_iter):
        with open(os.path.join(checkpoints_root, dataset_name, f"{ind}"), "wb") as store_file:
            pickle.dump((inputs.token_ids.cpu(), subword_mentions.ids.cpu(), subword_mentions.probs.cpu(),
                         inputs.eval_mask.cpu(), subword_mentions.dictionary, inputs.raw_mentions,
                         inputs.is_in_mention.cpu(), inputs.bioes.cpu()), store_file,
                        protocol=pickle.HIGHEST_PROTOCOL)
    return dataset_name


def postprocess_annotations(annotations, sentence):
    res = []
    for ann in annotations:
        begin_index = ann[0]
        end_index = ann[1]
        annotation = ann[2]
        requires_check = True
        while requires_check and end_index > begin_index:
            mention = sentence[begin_index:end_index]
            if mention.lower().endswith("\'s") and all([any([m in c for c in annotation[0].lower().split("_")])
                                                        for m in mention[:-2].lower().split()]) and not \
                    all([any([m in c for c in annotation[0].lower().split("_")]) for m in mention.lower().split()]):
                end_index -= 2
            elif mention[0] in string.punctuation or mention[0] == ' ':
                begin_index += 1
            elif mention[-1] in string.punctuation and mention.lower()[-4:] not in ["u.s.", "u.n."]:
                end_index -= 1
            elif mention[-1] == ' ':
                end_index -= 1
            elif mention.lower()[-3:] in ["u.s", "u.n"] and end_index < len(sentence) and sentence[end_index] == '.':
                end_index += 1
            elif mention.lower() in ["a", "the", "in", "out", "to", "of", "for", "at", "by", "rd", "th", "and",
                                     "or", "but", "on", "none", "is", "were", "was", "he", "she", "if", "as",
                                     "have", "had", "has", "who", "when", "where", "a lot", "a little", "here",
                                     "there", "\'s"]:
                end_index = begin_index
                requires_check = False
            else:
                requires_check = False
        if begin_index < end_index:
            res.append((begin_index, end_index, annotation))
    return res


def get_aida_set_phrase_splitted_documents(dataset_name):
    d_iter = AIDADataset().dataset[dataset_name]

    phrase_documents = []

    for document in d_iter:
        document_words = []
        document_labels = []
        document_candidates = []

        for annotation in document.annotations:
            for a in annotation:
                document_words.append(a.token)
                document_candidates.append([x.url.replace('http://en.wikipedia.org/wiki/', '')
                                            for x in a.candidates.candidates] if a.candidates else [])
                if a.yago_entity and a.yago_entity != "--NME--":
                    document_labels.append(a.yago_entity.encode('ascii').decode('unicode-escape'))
                else:
                    document_labels.append('|||O|||')
        original_string = " ".join(document_words)
        tokenized_mention = tokenizer(original_string)
        tokens_offsets = list(zip(tokenized_mention.tokens(), tokenized_mention.encodings[0].offsets))[1:-1]
        mapping = get_subword_to_word_mapping(tokenized_mention.tokens(), original_string)
        subword_tokens = tokenized_mention.tokens()[1:-1]
        w_ind = 0
        subword_annotations = []
        word_annotations = []
        for w, l, cnds in zip(document_words, document_labels, document_candidates):
            converted_to_words = "".join([x[1:] if x.startswith("\u0120")
                                          else x for x in subword_tokens[mapping[w_ind][0]:mapping[w_ind][1]]])
            if w == converted_to_words:
                for sub_w in subword_tokens[mapping[w_ind][0]:mapping[w_ind][1]]:
                    subword_annotations.append(SubwordAnnotation([1.0], [dl_sa.mentions_vocab[l]], sub_w))
                word_annotations.append(WordAnnotation(subword_annotations[mapping[w_ind][0]:mapping[w_ind][1]],
                                                       tokens_offsets[mapping[w_ind][0]:mapping[w_ind][1]], cnds))
                w_ind += 1
            elif len(mapping) > w_ind + 1 and w == "".join([x[1:] if x.startswith("\u0120")
                                                            else x for x in subword_tokens[
                                                                            mapping[w_ind][0]:mapping[w_ind+1][1]]]):
                for sub_w in subword_tokens[mapping[w_ind][0]:mapping[w_ind+1][1]]:
                    subword_annotations.append(SubwordAnnotation([1.0], [dl_sa.mentions_vocab[l]], sub_w))
                word_annotations.append(WordAnnotation(subword_annotations[mapping[w_ind][0]:mapping[w_ind+1][1]],
                                                       tokens_offsets[mapping[w_ind][0]:mapping[w_ind+1][1]], cnds))
                w_ind += 2
            else:
                raise ValueError("This should not happen")
        phrase_annotations = []
        for w in word_annotations:
            if phrase_annotations and phrase_annotations[-1].resolved_annotation == w.resolved_annotation:
                phrase_annotations[-1].add(w)
            else:
                phrase_annotations.append(PhraseAnnotation(w))
        phrase_documents.append(phrase_annotations)
    return phrase_documents


def _process_last_overlap(text_chunk_overlap, _overlap, l):
    """
    Function intended to merge the predictions in the text chunk overlaps.
    Implemented to be used in chunk_annotate_and_merge_to_phrase function.
    """
    if not l:
        l = _overlap
    if len(l) < len(_overlap):
        o = [x for x in _overlap]
        o[-len(l):] = l
        l = o
    _r = []
    if len(_overlap) < text_chunk_overlap:
        text_chunk_overlap = len(_overlap)
    for i in range(text_chunk_overlap):
        if _overlap[i] == 0:
            _r.append((l[i],))
        elif l[i] == 0 or _overlap[i] == l[i]:
            _r.append((_overlap[i],))
        else:  # keeping both for prediction resolution
            _r.append((l[i], _overlap[i]))
    return _r


def normalize_sentence_for_moses_alignment(sentence, normalize_for_chinese_characters=False):
    for k, v in [('\u2018', '\''), ('\u2019', '\''), ('\u201d', '\"'), ('\u201c', '\"'), ('\u2013', '-'),
                 ('\u2014', '-'), ('\u2026', '.'), ('\u2022', '.'), ('\u00f6', 'o'),('\u00e1', 'a'), ('\u00e8', 'e'),
                 ('\u00c9', 'E'), ('\u014d', 'o'), ('\u0219', 's'), ('\n', '\u010a'), ('\u00a0', ' '), ('\u694a', ' '),
                 ('\u9234', ' '), ('\u6797', ' '), ('\u6636', ' '), ('\u4f50', ' '), ('\u738b', ' '), ('\u5b9c', ' '),
                 ('\u6b63', ' '), ('\u5168', ' '), ('\u52dd', ' '), ('\u80e1', ' '), ('\u5fd7', ' '), ('\u535a', ' '),
                 ('\u9673', ' '), ('\u7f8e', ' '), ('\u20ac', 'E'), ('\u201e', '\"'), ('\u0107', 'c'), ('\ufeff', ' '),
                 ('\u017e', 'z'), ('\u010d', 'c')]:
        if k in sentence:
            sentence = sentence.replace(k, v)
    if normalize_for_chinese_characters:
        for k, v in [('\u5e7c', ' '), ('\u5049', ' '), ('\u5b8f', ' '), ('\u9054', ' '), ('\u5bb9', ' '),
                     ('\u96fb', ' '), ('\u590f', ' '), ('\u5b63', ' '), ('\u660c', ' '), ('\u90b1', ' '),
                     ('\u4fca', ' '), ('\u6587', ' '), ('\u56b4', ' '), ('\u5b87', ' '), ('\u67cf', ' '),
                     ('\u8b5a', ' '), ('\u9f0e', ' '), ('\u6176', ' '), ('\u99ac', ' '), ('\u82f1', ' '),
                     ('\u4e5d', ' '), ('\u6797', ' '), ('\u7537', ' '), ('\u9996', ' '), ('\u60e0', ' '),
                     ('\u7d00', ' '), ('\u5143', ' '), ('\u8f1d', ' '), ('\u5289', ' '), ('\u4fd0', ' '),
                     ('\u8208', ' '), ('\u4e2d', ' '), ('\u8b1d', ' '), ('\u5922', ' '), ('\u9e9f', ' '),
                     ('\u6e38', ' '), ('\u570b', ' '), ('\u7167', ' '), ('\u658c', ' '), ('\u54f2', ' '),
                     ('\u9ec3', ' '), ('\u5433', ' '), ('\u53cb', ' '), ('\u6e05', ' '), ('\u856d', ' '),
                     ('\u8000', ' '), ('\u5eb7', ' '), ('\u6dd1', ' '), ('\u83ef', ' ')]:
            if k in sentence:
                sentence = sentence.replace(k, v)
    return sentence


def chunk_annotate_and_merge_to_phrase(model, sentence, k_for_top_k_to_keep=5, normalize_for_chinese_characters=False):
    sentence = sentence.rstrip()
    sentence = normalize_sentence_for_moses_alignment(sentence, normalize_for_chinese_characters)
    simple_split_words = moses_tokenize(sentence)
    sentence = sentence.replace('\u010a', '\n')
    tokenized_mention = tokenizer(sentence)
    tokens_offsets = list(zip(tokenized_mention.tokens(), tokenized_mention.encodings[0].offsets))
    subword_to_word_mapping = get_subword_to_word_mapping(tokenized_mention.tokens(), sentence)
    chunks = [tokens_offsets[i: i + model.text_chunk_length] for i in range(
        0, len(tokens_offsets), model.text_chunk_length - model.text_chunk_overlap)]
    result = []
    last_overlap = []
    logits = []
    # ########################################################################################################
    # Covert each chunk to tensors, predict the labels, and merge the overlaps (keep conflicting predictions).
    # ########################################################################################################
    for chunk in chunks:
        subword_ids = [tokenizer.convert_tokens_to_ids([x[0] for x in chunk])]
        logits = model.annotate_subword_ids(
            subword_ids, k_for_top_k_to_keep, chunk)
        if last_overlap:
            result.extend(_process_last_overlap(model.text_chunk_overlap, last_overlap, logits))
        else:
            result.extend([(x,) for x in logits[:model.text_chunk_overlap]])
        if len(logits) > 2 * model.text_chunk_overlap:
            result.extend([(x,) for x in logits[model.text_chunk_overlap:-model.text_chunk_overlap]])
            last_overlap = logits[-model.text_chunk_overlap:]
        else:
            result.extend([(x,) for x in logits[model.text_chunk_overlap:]])
            last_overlap = []
        logits = []
    result.extend(_process_last_overlap(model.text_chunk_overlap, last_overlap, logits))
    # ########################################################################################################
    # Resolve the overlap merge conflicts using the model prediction probability
    # ########################################################################################################
    final_result = []
    for p_ind, prediction in enumerate(result):
        if len(prediction) == 1:
            final_result.append(prediction[0])
        else:
            p_found = False
            for p in prediction:
                if p == final_result[-1] or (p_ind + 1 < len(result) and p in result[p_ind + 1]):
                    # It is equal to the one in the left or in the one to the right
                    final_result.append(p)
                    p_found = True
                    break
            if not p_found:  # choose the one the model is more confident about
                final_result.append(sorted(prediction, key=lambda x: x.item_probability(), reverse=True)[0])
    # ########################################################################################################
    # Convert the model predictions (subword-level) to valid GERBIL annotation spans (continuous char-level)
    # ########################################################################################################
    tokens_offsets = tokens_offsets[1:-1]
    final_result = final_result[1:]
    # last_step_annotations = []
    word_annotations = [WordAnnotation(final_result[m[0]:m[1]], tokens_offsets[m[0]:m[1]])
                        for m in subword_to_word_mapping]
    # ########################################################################################################
    #                    MAKING SURE WORDS ARE NOT BROKEN IN SEPARATE PHRASES!
    # ########################################################################################################
    w_p_1 = 0
    w_p_2 = 0
    w_2_buffer = ""
    w_1_buffer = ""
    while w_p_1 < len(word_annotations) and w_p_2 < len(simple_split_words):
        w_1 = word_annotations[w_p_1]
        w_2 = normalize(simple_split_words[w_p_2]).strip()
        w_1_word_string = normalize(w_1.word_string).strip()
        if w_1_word_string == w_2:
            w_p_1 += 1
            w_p_2 += 1
        elif w_1_buffer and w_2_buffer and normalize(
                w_1_buffer + w_1.word_string).strip() == normalize(w_2_buffer + simple_split_words[w_p_2]).strip():
            w_p_1 += 1
            w_p_2 += 1
            w_1_buffer = ""
            w_2_buffer = ""
        elif w_2_buffer and w_1_word_string == normalize(w_2_buffer + simple_split_words[w_p_2]).strip():
            w_p_1 += 1
            w_p_2 += 1
            w_2_buffer = ""
        elif w_1_buffer and normalize(w_1_buffer + w_1.word_string).strip() == w_2:
            w_p_1 += 1
            w_p_2 += 1
            w_1_buffer = ""
        elif w_1_buffer and len(w_2) < len(normalize(w_1_buffer + w_1.word_string).strip()):
            w_2_buffer += simple_split_words[w_p_2]
            w_p_2 += 1
        elif len(w_2) < len(w_1_word_string):
            w_2_buffer += simple_split_words[w_p_2]
            w_p_2 += 1
        # Connecting the "." in between the names to the word it belongs to.
        elif len(w_2) > len(w_1_word_string) and w_p_1 + 1 < len(word_annotations) \
                and word_annotations[w_p_1 + 1].word_string == ".":
            word_annotations[w_p_1 + 1] = WordAnnotation(
                word_annotations[w_p_1].annotations + word_annotations[w_p_1 + 1].annotations,
                word_annotations[w_p_1].token_offsets + word_annotations[w_p_1 + 1].token_offsets)
            word_annotations[w_p_1].annotations = []
            word_annotations[w_p_1].token_offsets = []
            w_p_1 += 1
        elif len(w_2) > len(w_1_word_string) and w_p_1 + 1 < len(word_annotations):
            w_1_buffer += w_1.word_string
            w_p_1 += 1
        elif w_2_buffer and normalize(word_annotations[w_p_1].word_string + word_annotations[w_p_1 + 1].word_string).strip():
            w_p_1 += 2
            w_2_buffer = ""
        else:
            raise ValueError("This should not happen!")
    # ################################################################################################################
    phrase_annotations = []
    for w in word_annotations:
        if not w.annotations:
            continue
        if phrase_annotations and phrase_annotations[-1].resolved_annotation == w.resolved_annotation:
            phrase_annotations[-1].add(w)
        else:
            phrase_annotations.append(PhraseAnnotation(w))
    return phrase_annotations


class ComparisonResult(Enum):
    CORRECTLY_IGNORED_O = 0
    CORRECTLY_FOUND_BOTH_SPAN_AND_ANNOTATION = 1
    CORRECTLY_FOUND_SPAN_BUT_NOT_ANNOTATION = 2
    OVER_GENERATED_ANNOTATION = 3

    @staticmethod
    def get_correct_status(g_span, p_span):
        g_is_o = g_span.resolved_annotation == 0
        got_annotation_right = p_span.resolved_annotation == g_span.resolved_annotation
        got_span_right = p_span.word_string.replace(" ", "") == g_span.word_string.replace(" ", "")
        #  p_span.average_annotation_confidence == g_span.average_annotation_confidence
        if got_span_right and got_annotation_right and g_is_o:
            return ComparisonResult.CORRECTLY_IGNORED_O
        elif got_span_right and got_annotation_right and not g_is_o:
            return ComparisonResult.CORRECTLY_FOUND_BOTH_SPAN_AND_ANNOTATION
        elif got_span_right and not got_annotation_right and not g_is_o:
            # it could be that p is o or not!
            return ComparisonResult.CORRECTLY_FOUND_SPAN_BUT_NOT_ANNOTATION
        elif got_span_right and not got_annotation_right and g_is_o:
            return ComparisonResult.OVER_GENERATED_ANNOTATION
        else:
            raise ValueError("This should not happen!")


def compare_gold_and_predicted_annotation_documents(gold_document, predicted_document, ignore_over_generated=False,
                                                    ignore_predictions_outside_candidate_list=False):
    """
    Compares the output results of the model predictions and the gold annotations.
    """
    g_id = 0
    p_id = 0
    comparison_results = []
    while g_id < len(gold_document) and p_id < len(predicted_document):
        p_span = predicted_document[p_id]
        g_span = gold_document[g_id]
        special_condition = p_span.word_string != g_span.word_string and p_span.word_string.replace(
            " ", "") == g_span.word_string.replace(" ", "")
        if p_span.word_string == g_span.word_string or special_condition:
            p_id += 1
            g_id += 1
            comparison_results.append((g_span, p_span, ComparisonResult.get_correct_status(g_span, p_span)))
        elif len(p_span.word_string) < len(g_span.word_string) and \
                len(p_span.words) == len(g_span.words) == 1 and p_id + 1 < len(predicted_document) and \
                len(predicted_document[p_id+1].words) > 1:
            p_span.add(predicted_document[p_id+1].words[0])
            predicted_document[p_id+1].words.pop(0)
            continue
        elif len(p_span.word_string) < len(g_span.word_string):
            # potentially over-generated span later
            new_phrase = PhraseAnnotation(g_span.words[0])
            i = 1
            while new_phrase.word_string.replace(" ", "") != p_span.word_string.replace(" ", "") \
                    and i < len(g_span.words):
                new_phrase.add(g_span.words[i])
                i += 1
            not_solved = new_phrase.word_string.replace(" ", "") != p_span.word_string.replace(" ", "")
            if not_solved and p_id + 1 < len(predicted_document) and len(predicted_document[p_id+1].words) > 1:
                p_span.add(predicted_document[p_id+1].words[0])
                predicted_document[p_id+1].words.pop(0)
                continue
            elif not_solved and p_id + 1 < len(predicted_document) and len(predicted_document[p_id+1].words) == 1:
                p_span.add(predicted_document[p_id+1].words[0])
                predicted_document[p_id+1].words = p_span.words
                predicted_document[p_id+1].set_alternative_as_resolved_annotation(p_span.resolved_annotation)
                p_id += 1
                continue
            elif not_solved:
                raise ValueError("This should not happen!")
            else:
                comparison_results.append((
                    new_phrase, p_span, ComparisonResult.get_correct_status(new_phrase, p_span)))
                g_span.words = g_span.words[i:]
                p_id += 1
        elif len(p_span.word_string) > len(g_span.word_string):
            # potentially missed a span
            new_phrase = PhraseAnnotation(p_span.words[0])
            i = 1
            while new_phrase.word_string.replace(" ", "") != g_span.word_string.replace(" ", "") \
                    and i < len(p_span.words):
                new_phrase.add(p_span.words[i])
                i += 1
            if new_phrase.word_string.replace(" ", "") != g_span.word_string.replace(" ", ""):
                # re-alignment not helpful
                new_p = PhraseAnnotation(p_span.words[0])
                new_g = PhraseAnnotation(g_span.words[0])
                i = 1
                while new_p.word_string == new_g.word_string:
                    new_p.add(p_span.words[i])
                    new_g.add(g_span.words[i])
                    i += 1
                new_p.words = new_p.words[:-1]
                new_g.words = new_g.words[:-1]
                comparison_results.append((new_g, new_p, ComparisonResult.get_correct_status(new_g, new_p)))
                p_span.words = p_span.words[i - 1:]
                g_span.words = g_span.words[i - 1:]
            else:
                comparison_results.append((
                    g_span, new_phrase, ComparisonResult.get_correct_status(g_span, new_phrase)))
                p_span.words = p_span.words[i:]
                g_id += 1
        elif g_span.word_string.replace(" ", "").startswith(p_span.word_string.replace(" ", "")) and \
                p_id + 1 < len(predicted_document) and p_span.word_string.replace(" ", "") + \
                predicted_document[p_id + 1].word_string.replace(" ", "") == g_span.word_string.replace(" ", ""):
            for next_span_word in predicted_document[p_id+1].words:
                p_span.add(next_span_word)
            predicted_document[p_id+1] = p_span
            p_id += 1
            continue
        elif g_span.word_string.replace(" ", "").startswith(p_span.word_string.replace(" ", "")) and \
                p_id + 1 < len(predicted_document) and p_span.word_string.replace(" ", "") + \
                predicted_document[p_id + 1].words[0].word_string.replace(" ", "") == \
                g_span.word_string.replace(" ", ""):
            p_span.add(predicted_document[p_id+1].words[0])
            predicted_document[p_id+1].words.pop(0)
            continue
        elif g_span.word_string.replace(" ", "").startswith(p_span.word_string.replace(" ", "")):
            raise ValueError("This should be handled correctly!")
        elif p_span.word_string.replace(" ", "").startswith(g_span.word_string.replace(" ", "")):
            raise ValueError("This should be handled correctly!")
        else:
            raise ValueError("This should not happen!")
    if ignore_over_generated:
        c_results = []
        for g, p, r in comparison_results:
            if ignore_over_generated and r == ComparisonResult.OVER_GENERATED_ANNOTATION:
                p.set_alternative_as_resolved_annotation(0)
                r = ComparisonResult.CORRECTLY_IGNORED_O
            c_results.append((g, p, r))
        comparison_results = c_results
    if ignore_predictions_outside_candidate_list:
        c_results = []
        for g, p, r in comparison_results:
            g_ppr_for_ned_candidates = [dl_sa.mentions_vocab[x] for x in g.ppr_for_ned_candidates if x in dl_sa.mentions_vocab]
            if g_ppr_for_ned_candidates:
                all_p_anns = p.all_possible_annotations()
                filtered_p_predictions = sorted([x for x in all_p_anns if x[0] in g_ppr_for_ned_candidates],
                                                key=lambda y: y[1], reverse=True)
                if filtered_p_predictions:
                    p.set_alternative_as_resolved_annotation(filtered_p_predictions[0][0])
                else:
                    p.set_alternative_as_resolved_annotation(0)
                r = ComparisonResult.get_correct_status(g, p)
            c_results.append((g, p, r))
        comparison_results = c_results
    return comparison_results
