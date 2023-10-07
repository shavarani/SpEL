import os
import torch
import pathlib
import json
from datetime import date

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

AIDA_CANONICAL_REDIRECTS = None
OOD_CANONICAL_REDIRECTS = None


def get_base_model_name():
    return open("base_model.cfg", "r").read().strip()


def get_project_top_dir():
    return pathlib.Path(os.path.abspath(__file__)).parent.parent.parent


def get_resources_dir():
    return get_project_top_dir() / 'resources'


def get_checkpoints_dir():
    path_ = get_project_top_dir() / '.checkpoints'
    if not os.path.exists(path_):
        os.mkdir(path_)
    return path_


def get_logdir_dir():
    path_ = get_project_top_dir() / '.logdir'
    if not os.path.exists(path_):
        os.mkdir(path_)
    return path_


def get_aida_train_canonical_redirects():
    global AIDA_CANONICAL_REDIRECTS
    if not AIDA_CANONICAL_REDIRECTS:
        r_file = get_resources_dir() / "data" / "aida_canonical_redirects.json"
        with r_file.open() as f:
            AIDA_CANONICAL_REDIRECTS = json.load(f)
    return AIDA_CANONICAL_REDIRECTS

def get_ood_canonical_redirects():
    global OOD_CANONICAL_REDIRECTS
    if not OOD_CANONICAL_REDIRECTS:
        r_file = get_resources_dir() / "data" / "ood_canonical_redirects.json"
        with r_file.open() as f:
            OOD_CANONICAL_REDIRECTS = json.load(f)
    return OOD_CANONICAL_REDIRECTS


def get_aida_yago_tsv_file_path():
    return get_resources_dir() / "data" / "AIDA-YAGO2-dataset.tsv"


def get_exec_run_file():
    return get_logdir_dir() / f"annotator_log-{date.today().strftime('%Y-%b-%d')}.log"


def get_aida_vocab():
    mentions_vocab = dict({'|||O|||': 0, '<pad>': 1})
    dictionary_file = get_resources_dir() / "vocab" / "aida.txt"
    dfile = dictionary_file.open("r")
    for _ad_element in dfile.read().split("\n"):
        mentions_vocab[_ad_element] = len(mentions_vocab)
    return mentions_vocab

def get_ood_vocab():
    # This function might be used if one is interested in testing out the "masking all the candidates not in our
    #   expected entity set" which is mentioned in the footnote of section 4.1 of the paper.
    mentions_vocab = dict({'|||O|||': 0, '<pad>': 1})
    dictionary_file = get_resources_dir() / "vocab" / "out_of_domain.txt"
    dfile = dictionary_file.open("r")
    for _ad_element in dfile.read().split("\n"):
        mentions_vocab[_ad_element] = len(mentions_vocab)
    return mentions_vocab


def get_aida_plus_wikipedia_vocab():
    mentions_vocab = get_aida_vocab()
    dictionary_file = get_resources_dir() / "vocab" / f"enwiki_20230827.txt"
    dfile = dictionary_file.open("r")
    for _ad_element in dfile.read().split("\n"):
        if _ad_element not in mentions_vocab:
            mentions_vocab[_ad_element] = len(mentions_vocab)
    return mentions_vocab

def get_aida_plus_wikipedia_plus_out_of_domain_vocab():
    mentions_vocab = get_aida_plus_wikipedia_vocab()
    dictionary_file = get_resources_dir() / "vocab" / f"out_of_domain.txt"
    dfile = dictionary_file.open("r")
    for _ad_element in dfile.read().split("\n"):
        if _ad_element not in mentions_vocab:
            mentions_vocab[_ad_element] = len(mentions_vocab)
    return mentions_vocab

def get_n3_entity_to_kb_mappings():
    kb_file = get_resources_dir() / "data" / "n3_kb_mappings.json"
    knowledge_base = json.load(kb_file.open("r"))
    return knowledge_base
