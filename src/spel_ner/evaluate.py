"""
Content of this file is borrowed from https://github.com/asahi417/tner for comparability of the comparisons.
"""
import os
import logging
import requests
import json
import hashlib
from typing import List
import numpy as np
from itertools import chain
from tqdm import tqdm
from scipy.stats import bootstrap
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

def get_shared_label(cache_dir: str = None):
    """ universal label set to unify the NER datasets

    @param cache_dir: cache directly
    @return: a dictionary mapping from label to id
    """
    cache_dir = '.' if cache_dir is None else cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    url = f"https://raw.githubusercontent.com/asahi417/tner/master/unified_label2id.json"
    path = os.path.join(cache_dir, "unified_label2id.json")
    if os.path.exists(path):
        with open(path, 'rb') as f:
            checksum = hashlib.md5(f.read()).hexdigest()
        if 'a6b6bbfd6ddf3f990ee6b335ade46429' == checksum:
            with open(path) as f:
                label2id = json.load(f)
            return label2id
        else:
            logging.warning('local `unified_label2id.json` has wrong checksum')
    with open(path, "w") as f:
        logging.info(f'downloading `unified_label2id.json` from {url}')
        r = requests.get(url)
        label2id = json.loads(r.content)
        json.dump(label2id, f)
    return label2id

def span_f1(pred_list: List[List],
            label_list: List[List],
            span_detection_mode: bool = False,
            return_ci: bool = False,
            unification_by_shared_label: bool = True):
    """ calculate span F1 score

    @param pred_list: a list of predicted tag sequences
    @param label_list: a list of gold tag sequences
    @param return_ci: [optional] return confidence interval by bootstrap
    @param span_detection_mode: [optional] return F1 of entity span detection (ignoring entity type error and cast
        as binary sequence classification as below)
        - NER                  : ["O", "B-PER", "I-PER", "O", "B-LOC", "O", "B-ORG"]
        - Entity-span detection: ["O", "B-ENT", "I-ENT", "O", "B-ENT", "O", "B-ENT"]
    @param unification_by_shared_label: [optional] map entities into a shared form
    @return: a dictionary containing span f1 scores
    """
    if unification_by_shared_label:
        unified_label_set = get_shared_label()
        logging.info(f'map entity into shared label set {unified_label_set}')

        def convert_to_shared_entity(entity_label):
            if entity_label == 'O':
                return entity_label
            prefix = entity_label.split('-')[0]  # B or I
            entity = '-'.join(entity_label.split('-')[1:])
            normalized_entity = [k for k, v in unified_label_set.items() if entity in v]
            assert len(
                normalized_entity) <= 1, f'duplicated entity found in the shared label set\n {normalized_entity} \n {entity}'
            if len(normalized_entity) == 0:
                logging.warning(f'Entity `{entity}` is not found in the shared label set {unified_label_set}. '
                                f'Original entity (`{entity}`) will be used as label.')
                return f'{prefix}-{entity}'
            return f'{prefix}-{normalized_entity[0]}'


        label_list = [[convert_to_shared_entity(_i) for _i in i] for i in label_list]
        pred_list = [[convert_to_shared_entity(_i) for _i in i] for i in pred_list]

    if span_detection_mode:
        logging.info(f'span_detection_mode: map entity into binary label set')

        def convert_to_binary_mask(entity_label):
            if entity_label == 'O':
                return entity_label
            prefix = entity_label.split('-')[0]  # B or I
            return f'{prefix}-entity'

        label_list = [[convert_to_binary_mask(_i) for _i in i] for i in label_list]
        pred_list = [[convert_to_binary_mask(_i) for _i in i] for i in pred_list]

    # compute metrics
    logging.info(f'\n{classification_report(label_list, pred_list)}')
    m_micro, ci_micro = span_f1_single(label_list, pred_list, average='micro', return_ci=return_ci)
    m_macro, ci_macro = span_f1_single(label_list, pred_list, average='macro', return_ci=return_ci)
    metric = {
        "micro/f1": m_micro,
        "micro/f1_ci": ci_micro,
        "micro/recall": recall_score(label_list, pred_list, average='micro'),
        "micro/precision": precision_score(label_list, pred_list, average='micro'),
        "macro/f1": m_macro,
        "macro/f1_ci": ci_macro,
        "macro/recall": recall_score(label_list, pred_list, average='macro'),
        "macro/precision": precision_score(label_list, pred_list, average='macro'),
    }
    target_names = sorted(list(set([k.replace('B-', '') for k in list(chain(*label_list)) if k.startswith('B-')])))

    if not span_detection_mode:
        metric["per_entity_metric"] = {}
        for t in target_names:
            _label_list = [[_i if _i.endswith(t) else 'O' for _i in i] for i in label_list]
            _pred_list = [[_i if _i.endswith(t) else 'O' for _i in i] for i in pred_list]
            m, ci = span_f1_single(_label_list, _pred_list, return_ci=return_ci)
            metric["per_entity_metric"][t] = {
                "f1": m,
                "f1_ci": ci,
                "precision": precision_score(_label_list, _pred_list),
                "recall": recall_score(_label_list, _pred_list)}
    return metric

def span_f1_single(label_list,
                   pred_list,
                   random_seed: int = 0,
                   n_resamples: int = 1000,
                   confidence_level: List = None,
                   return_ci: bool = False,
                   average: str = 'macro'):
    """ span-F1 score with bootstrap CI (data.shape == (n_sample, 2)) """
    data = np.array(list(zip(pred_list, label_list)), dtype=object)

    def get_f1(xy, axis=None):
        assert len(xy.shape) in [2, 3], xy.shape
        prediction = xy[0]
        label = xy[1]
        if axis == -1 and len(xy.shape) == 3:
            assert average is not None
            tmp = []
            for i in tqdm(list(range(len(label)))):
                tmp.append(f1_score(label[i, :], prediction[i, :], average=average))
            return np.array(tmp)
        assert average is not None
        return f1_score(label, prediction, average=average)

    confidence_level = confidence_level if confidence_level is not None else [90, 95]
    mean_score = get_f1(data.T)
    ci = {}
    if return_ci:
        for c in confidence_level:
            logging.info(f'computing confidence interval: {c}')
            res = bootstrap(
                (data,),
                get_f1,
                confidence_level=c * 0.01,
                method='percentile',
                n_resamples=n_resamples,
                random_state=np.random.default_rng(random_seed)
            )
            ci[str(c)] = [res.confidence_interval.low, res.confidence_interval.high]
    return mean_score, ci