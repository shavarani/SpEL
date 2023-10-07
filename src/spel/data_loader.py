"""
The enwiki/conll Dataset reader/provider using torchtext.
The datasets were crated using the scripts from:
    https://github.com/samuelbroscheit/entity_knowledge_in_bert/tree/master/bert_entity/preprocessing
The get_dataset.collate_batch function is influenced by:
    https://raw.githubusercontent.com/samuelbroscheit/entity_knowledge_in_bert/master/bert_entity/data_loader_wiki.py

Please note that the pre-processed fine-tuning data will be automatically downloaded upon instantiation of the data
 readers and the result will be saved under /home/<user_name>/.cache/torch/text/datasets/ (in linux systems)

The expected sizes of the auto-downloaded datasets:
    - Step 1 (general knowledge fine-tuning):
            enwiki-2023-spel-roberta-tokenized-aug-27-2023.tar.gz: 19.1 GBs
            * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
            * You can delete the file above once fine-tuning step 1 is done, and you are moving on to step 2.         *
            * in the cleaning up process, make sure you remove the cached validation set files under .checkpoints     *
            * directory as well                                                                                       *
            * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    - Step 2 (general knowledge fine-tuning):
            enwiki-2023-spel-roberta-tokenized-aug-27-2023-retokenized.tar.gz: 17.5 GBs
            * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
            * You can delete the file above once fine-tuning step 2 is done, and you are moving on to step 3.         *
            * in the cleaning up process, make sure you remove the cached validation set files under .checkpoints     *
            * directory as well                                                                                       *
            * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    - Step 3 (domain specific fine-tuning):
            aida-conll-spel-roberta-tokenized-aug-23-2023.tar.gz: 5.1 MBs

No extra preprocessing step will be required, as soon as you start the fine-tuning script for each step,
 the proper fine-tuning dataset will be downloaded and will be served **without** the need for unzipping.
"""
import os
import json
import numpy
from functools import partial
from collections import OrderedDict
from tqdm import tqdm
from typing import Union, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchdata.datapipes.iter import FileOpener, IterableWrapper, HttpReader, FileLister
from torchtext.data.datasets_utils import _wrap_split_argument, _create_dataset_directory
from torchtext.utils import download_from_url

from transformers import AutoTokenizer, BatchEncoding

from spel.configuration import (get_aida_plus_wikipedia_plus_out_of_domain_vocab, get_aida_train_canonical_redirects,
                                get_aida_vocab, get_ood_vocab, get_checkpoints_dir, get_base_model_name, device)

BERT_MODEL_NAME = get_base_model_name()
MAX_SPAN_ANNOTATION_SIZE = 4


class StaticAccess:
    def __init__(self):
        self.mentions_vocab, self.mentions_itos = None, None
        self.set_vocab_and_itos_to_all()
        self.aida_canonical_redirects = get_aida_train_canonical_redirects()
        self._all_vocab_mask_for_aida = None
        self._all_vocab_mask_for_ood = None

    def set_vocab_and_itos_to_all(self):
        self.mentions_vocab = get_aida_plus_wikipedia_plus_out_of_domain_vocab()
        self.mentions_itos = [w[0] for w in sorted(self.mentions_vocab.items(), key=lambda x: x[1])]

    @staticmethod
    def get_aida_vocab_and_itos():
        aida_mentions_vocab = get_aida_vocab()
        aida_mentions_itos = [w[0] for w in sorted(aida_mentions_vocab.items(), key=lambda x: x[1])]
        return aida_mentions_vocab, aida_mentions_itos

    def shrink_vocab_to_aida(self):
        self.mentions_vocab, self.mentions_itos = self.get_aida_vocab_and_itos()

    def get_all_vocab_mask_for_aida(self):
        if self._all_vocab_mask_for_aida is None:
            mentions_vocab = get_aida_plus_wikipedia_plus_out_of_domain_vocab()
            mask = torch.ones(len(mentions_vocab)).to(device)
            mask = mask * -10000
            mask[torch.Tensor([mentions_vocab[x] for x in get_aida_vocab()]).long()] = 0
            self._all_vocab_mask_for_aida = mask
        return self._all_vocab_mask_for_aida

    def get_all_vocab_mask_for_ood(self):
        if self._all_vocab_mask_for_ood is None:
            mentions_vocab = get_aida_plus_wikipedia_plus_out_of_domain_vocab()
            mask = torch.ones(len(mentions_vocab)).to(device)
            mask = mask * -10000
            mask[torch.Tensor([mentions_vocab[x] for x in get_ood_vocab()]).long()] = 0
            self._all_vocab_mask_for_ood = mask
        return self._all_vocab_mask_for_ood


dl_sa = StaticAccess()


class ENWIKI20230827Config:
    URL = "https://1sfu-my.sharepoint.com/:u:/g/personal/sshavara_sfu_ca/Ea3IVbOpkTJKpASNyL9aFGMBQpH0ABU2hQa-wYyakkZ9TQ?e=DJFF3v&download=1"
    MD5 = "eb9a54a8f1f858cdcbf6c750942a896f"
    PATH = "enwiki-2023-spel-roberta-tokenized-aug-27-2023.tar.gz"
    DATASET_NAME = "WIKIPEDIA20230827"
    NUM_LINES = {'train': 3055221, 'valid': 1000, 'test': 1000}


class ENWIKI20230827V2Config:
    URL = 'https://1sfu-my.sharepoint.com/:u:/g/personal/sshavara_sfu_ca/EeS_Tgl_CFJNiTh6YH5IDrsBocEZUsZV3lxPB6pleTxyxw?e=caH1cf&download=1'
    MD5 = "83a37f528800a463cd1a376e80ffc744"
    PATH = "enwiki-2023-spel-roberta-tokenized-aug-27-2023-retokenized.tar.gz"
    DATASET_NAME = "WIKIPEDIA20230827V2"
    NUM_LINES = {'train': 3038581, 'valid': 996}


class AIDA20230827Config:
    URL = "https://1sfu-my.sharepoint.com/:u:/g/personal/sshavara_sfu_ca/EajEGYyf8LBOoxqDaiPBvbgBwFuEC08nssvZwGJWsG_HXg?e=wAwV6H&download=1"
    MD5 = "8078529d5df96d0d1ecf6a505fdb767a"
    PATH = "aida-conll-spel-roberta-tokenized-aug-23-2023.tar.gz"
    DATASET_NAME = "AIDA20230827"
    NUM_LINES = {'train': 1585, 'valid': 391, 'test': 372}


tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME, cache_dir=get_checkpoints_dir() / "hf")

WIKI_EXTRACTED_FILES = {"train": "train.json", "valid": "valid.json", "test": " test.json"}


def wiki_filter_fn(split, fname_and_stream):
    return WIKI_EXTRACTED_FILES[split] in fname_and_stream[0]


def wiki_data_record_convert(line):
    element = json.loads(line)
    r = {'tokens': element['tokens'], 'mentions': [], 'mention_entity_probs': [], 'mention_probs': []}
    for token, mentions, mention_entity_probs, mention_probs in zip(element['tokens'], element['mentions'],
                                                                    element['mention_entity_probs'],
                                                                    element['mention_probs']):
        if len(mention_probs) < len(mentions):
            mention_probs.extend([1.0 for _ in range(len(mentions) - len(mention_probs))])
        sorted_mentions = sorted(list(zip(mentions, mention_entity_probs, mention_probs)),
                                 key=lambda x: x[1], reverse=True)
        mentions_ = [dl_sa.aida_canonical_redirects[x[0]] if x[0] in dl_sa.aida_canonical_redirects else x[0]
                     for x in sorted_mentions if x[0]]  # ignore mentions that are None
        mention_entity_probs_ = [x[1] for x in sorted_mentions if x[0]]  # ignore prob. for None mentions
        mention_probs_ = [x[2] for x in sorted_mentions if x[0]]  # ignore m_probs for None mentions
        r['mentions'].append(mentions_[:MAX_SPAN_ANNOTATION_SIZE])
        r['mention_probs'].append(mention_probs_[:MAX_SPAN_ANNOTATION_SIZE])
        r['mention_entity_probs'].append(mention_entity_probs_[:MAX_SPAN_ANNOTATION_SIZE])
        if len(mentions_) > MAX_SPAN_ANNOTATION_SIZE:
            r['mention_entity_probs'][-1] = [x / sum(r['mention_entity_probs'][-1])
                                             for x in r['mention_entity_probs'][-1]]
    return r

@_create_dataset_directory(dataset_name=ENWIKI20230827Config.DATASET_NAME)
@_wrap_split_argument(("train", "valid", "test"))
def ENWIKI20230827(root: str, split: Union[Tuple[str], str]):
    root = root if root else ".data"
    path = root + "/" + ENWIKI20230827Config.PATH
    if not os.path.exists(path):
        download_from_url(ENWIKI20230827Config.URL, root=root, path=path, hash_value=ENWIKI20230827Config.MD5,
                          hash_type='md5')
    online_reader_dp = FileLister(root, ENWIKI20230827Config.PATH)
    tar_file_dp = FileOpener(online_reader_dp, mode="b").load_from_tar().filter(
        partial(wiki_filter_fn, split)).readlines(return_path=False).map(wiki_data_record_convert)
    return tar_file_dp

@_create_dataset_directory(dataset_name=ENWIKI20230827V2Config.DATASET_NAME)
@_wrap_split_argument(("train", "valid"))
def ENWIKI20230827V2(root: str, split: Union[Tuple[str], str]):
    root = root if root else ".data"
    path = root + "/" + ENWIKI20230827V2Config.PATH
    if not os.path.exists(path):
        download_from_url(ENWIKI20230827V2Config.URL, root=root, path=path, hash_value=ENWIKI20230827V2Config.MD5,
                          hash_type='md5')
    online_reader_dp = FileLister(root, ENWIKI20230827V2Config.PATH)
    tar_file_dp = FileOpener(online_reader_dp, mode="b").load_from_tar().filter(
        partial(wiki_filter_fn, split)).readlines(return_path=False).map(wiki_data_record_convert)
    return tar_file_dp

def aida_path_fn(r, _=None):
    return os.path.join(r, AIDA20230827Config.PATH)


def aida_select_split(s, file_name_data):
    return file_name_data[1][s]


def aida_data_record_convert(r):
    for x in r:  # making sure each token comes with exactly one annotation
        assert len(x) == 7 or len(x) == 8  # whether it contains the candidates or not
        return {"tokens": [x[0] for x in r], "mentions": [[x[4] if x[4] else "|||O|||"] for x in r],
                "mention_entity_probs": [[1.0] for _ in r], "mention_probs": [[1.0] for _ in r],
                "candidates": [x[7] if x[7] else [] for x in r] if len(x) == 8 else [[] for x in r]}


@_create_dataset_directory(dataset_name=AIDA20230827Config.DATASET_NAME)
@_wrap_split_argument(('train', 'valid', 'test'))
def AIDA20230827(root, split):
    online_reader_dp = HttpReader(IterableWrapper([AIDA20230827Config.URL])).on_disk_cache(
        filepath_fn=partial(aida_path_fn, root), hash_dict={aida_path_fn(root): AIDA20230827Config.MD5},
        hash_type="md5").end_caching(mode="wb", same_filepath_fn=True)
    return FileOpener(online_reader_dp, mode="b").load_from_tar().parse_json_files().flatmap(
        partial(aida_select_split, split)).map(aida_data_record_convert)


class DistributableDataset(Dataset):
    """
    Based on the documentations in torch.utils.data.DataLoader, `IterableDataset` does not support custom `sampler`
    Therefore we cannot use the DistributedSampler with the DataLoader to split the data samples.
    This class is a workaround to make the IterableDataset work with the DistributedSampler.
    """
    def __init__(self, dataset, size, world_size, rank):
        self.size = size
        self.data = iter(dataset)
        self.world_size = world_size
        self.rank = rank
        self.initial_fetch = True

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # Since we don't have a means of accessing the data by indices, we try skipping the indices that we believe
        #  belong to other processes
        skip_size = self.rank if self.initial_fetch else self.world_size - 1
        self.initial_fetch = False
        for _ in range(skip_size):
            next(self.data)
        return next(self.data)


def convert_is_in_mention_to_bioes(is_in_mention):
    # B = 0, I = 1, O = 2, E = 3, S = 4
    bioes = []
    for iim, current in enumerate(is_in_mention):
        before = is_in_mention[iim - 1] if iim > 0 else 0
        after = is_in_mention[iim + 1] if iim < len(is_in_mention) - 1 else 0
        bioes.append(
            2 if not current else (4 if not before and not after else (0 if not before else (3 if not after else 1))))
    return bioes


def get_dataset(dataset_name: str, split: str, batch_size: int, get_labels_with_high_model_score=None,
                label_size: int = 0, load_distributed: bool = False, world_size: int = 1, rank: int = 0,
                use_retokenized_wikipedia_data: bool = False):
    """
    :param dataset_name: The dataset name can either be "enwiki" or "aida"
    :param split: the requested dataset split which can be 'train', 'valid' or 'test'
    :param batch_size: the size of the resulting batch from the data loader
    :param get_labels_with_high_model_score: The function that finds high scoring negative samples for the model
    :param label_size: The maximum output distribution size. You can pass the output vocabulary size for this parameter.
    :param load_distributed: The flag hinting whether the data loader will be loaded in a multi-gpu setting.
    :param world_size: the number of machines that the dataloader is expected to serve.
    :param rank: the rank of the gpu on which the data is expected to be served.
    :param use_retokenized_wikipedia_data: a flag indicating whether to use ENWIKI20230827 dataset or ENWIKI20230827V2
    """

    assert dataset_name in ["enwiki", "aida"]
    if not load_distributed or rank == 0:
        print(f"Initializing the {dataset_name.upper()}/{split} dataset ...")

    def collate_batch(batch):
        data = {}
        for key in ["tokens", "mentions", "mention_entity_probs", "eval_mask", "candidates", "is_in_mention", "bioes"]:
            data[key] = []
        for annotated_line_in_file in batch:
            data["tokens"].append(tokenizer.convert_tokens_to_ids(annotated_line_in_file["tokens"]))
            data["mentions"].append([
                [(dl_sa.mentions_vocab[x] if x not in dl_sa.aida_canonical_redirects else
                  dl_sa.mentions_vocab[dl_sa.aida_canonical_redirects[x]])
                 if x is not None and x not in ['Gmina_Żabno'] else dl_sa.mentions_vocab["|||O|||"] for x in el]
                for el in annotated_line_in_file["mentions"]
            ])
            data["mention_entity_probs"].append(annotated_line_in_file["mention_entity_probs"])
            data["eval_mask"].append(list(map(
                lambda item: 1 if len(item) == 1 else 0, annotated_line_in_file["mention_probs"])))
            is_in_mention = [1 if x != '|||O|||' else 0 for el, elp in zip(
                annotated_line_in_file["mentions"], annotated_line_in_file["mention_entity_probs"])
                             for x, y in zip(el, elp) if y == max(elp)]
            data["is_in_mention"].append(is_in_mention)
            data["bioes"].append(convert_is_in_mention_to_bioes(is_in_mention))

        maxlen = max([len(x) for x in data["tokens"]])
        token_ids = torch.LongTensor([sample + [0] * (maxlen - len(sample)) for sample in data["tokens"]])
        eval_mask = torch.LongTensor([sample + [0] * (maxlen - len(sample)) for sample in data["eval_mask"]])
        is_in_mention = torch.LongTensor([sample + [0] * (maxlen - len(sample)) for sample in data["is_in_mention"]])
        bioes = torch.LongTensor([sample + [2] * (maxlen - len(sample)) for sample in data["bioes"]])
        if get_labels_with_high_model_score:
            labels_with_high_model_score = get_labels_with_high_model_score(token_ids)
        else:
            labels_with_high_model_score = None
        subword_mentions = create_output_with_negative_examples(
            data["mentions"], data["mention_entity_probs"], token_ids.size(0), token_ids.size(1),
            len(dl_sa.mentions_vocab), label_size, labels_with_high_model_score)
        inputs = BatchEncoding({
            'token_ids': token_ids,
            'eval_mask': eval_mask,
            'raw_mentions': data["mentions"],
            'is_in_mention': is_in_mention,
            "bioes": bioes
        })
        return inputs, subword_mentions
    if not load_distributed or rank == 0:
        print(f"Done initializing the {dataset_name.upper()}/{split} dataset ...")
    wikipedia_dataset = ENWIKI20230827
    wikipedia_dataset_config = ENWIKI20230827Config
    retokenized_wikipedia_dataset = ENWIKI20230827V2
    retokenized_wikipedia_dataset_config = ENWIKI20230827V2Config
    aida_dataset = AIDA20230827
    aida_dataset_config = AIDA20230827Config
    dset_class = (retokenized_wikipedia_dataset if use_retokenized_wikipedia_data else wikipedia_dataset) \
        if dataset_name == "enwiki" else aida_dataset
    d_size = (retokenized_wikipedia_dataset_config.NUM_LINES[split] if use_retokenized_wikipedia_data else
              wikipedia_dataset_config.NUM_LINES[split]) \
        if dataset_name == "enwiki" else aida_dataset_config.NUM_LINES[split]
    dataset_ = DistributableDataset(dset_class(split=split, root=get_checkpoints_dir()), d_size, world_size, rank) \
        if load_distributed else dset_class(split=split, root=get_checkpoints_dir())
    return DataLoader(dataset_, batch_size=batch_size, collate_fn=collate_batch,
                      sampler=DistributedSampler(dataset_, num_replicas=world_size, rank=rank)) \
        if load_distributed and split == "train" else DataLoader(dset_class(split=split, root=get_checkpoints_dir()),
                                                                 batch_size=batch_size,
                                                                 collate_fn=collate_batch)


def create_output_with_negative_examples(batch_entity_ids, batch_entity_probs, batch_size, maxlen, label_vocab_size,
                                         label_size, labels_with_high_model_score=None):
    all_entity_ids = OrderedDict()
    for batch_offset, (batch_item_token_item_entity_ids, batch_item_token_entity_probs) in enumerate(
            zip(batch_entity_ids, batch_entity_probs)
    ):
        for tok_id, (token_entity_ids, token_entity_probs) in enumerate(
                zip(batch_item_token_item_entity_ids, batch_item_token_entity_probs)
        ):
            for eid in token_entity_ids:
                if eid not in all_entity_ids:
                    all_entity_ids[eid] = len(all_entity_ids)
    # #####################################################
    shared_label_ids = list(all_entity_ids.keys())

    if len(shared_label_ids) < label_size and labels_with_high_model_score is not None:
        negative_examples = set(labels_with_high_model_score)
        negative_examples.difference_update(shared_label_ids)
        shared_label_ids += list(negative_examples)

    if len(shared_label_ids) < label_size:
        negative_samples = set(numpy.random.choice(label_vocab_size, label_size, replace=False))
        negative_samples.difference_update(shared_label_ids)
        shared_label_ids += list(negative_samples)

    shared_label_ids = shared_label_ids[: label_size]

    all_batch_entity_ids, batch_shared_label_ids = all_entity_ids, shared_label_ids
    if label_size > 0:
        label_probs = torch.zeros(batch_size, maxlen, len(batch_shared_label_ids))
    else:
        label_probs = torch.zeros(batch_size, maxlen, label_vocab_size)
    # loop through the batch x tokens x (label_ids, label_probs)
    for batch_offset, (batch_item_token_item_entity_ids, batch_item_token_entity_probs) in enumerate(
            zip(batch_entity_ids, batch_entity_probs)
    ):
        # loop through tokens x (label_ids, label_probs)
        for tok_id, (token_entity_ids, token_entity_probs) in enumerate(
                zip(batch_item_token_item_entity_ids, batch_item_token_entity_probs)):
            if label_size is None:
                label_probs[batch_offset][tok_id][torch.LongTensor(token_entity_ids)] = torch.Tensor(
                    batch_item_token_item_entity_ids)
            else:
                label_probs[batch_offset][tok_id][
                    torch.LongTensor(list(map(all_batch_entity_ids.__getitem__, token_entity_ids)))
                ] = torch.Tensor(token_entity_probs)

    label_ids = torch.LongTensor(batch_shared_label_ids)
    return BatchEncoding({
        "ids": label_ids,  # of size label_size
        "probs": label_probs,  # of size input_batch_size x input_max_len x label_size
        "dictionary": {v: k for k, v in all_batch_entity_ids.items()}  # contains all original ids for mentions in batch
    })


def _make_vocab_file():
    wiki_vocab = set()
    vocab_file = open("enwiki_20230827.txt", "w")
    for spl in ['train', 'valid', 'test']:
        for el in tqdm(ENWIKI20230827(split=spl, root=get_checkpoints_dir())):
            for x in el['mentions']:
                for y in x:
                    if y not in wiki_vocab:
                        vocab_file.write(f"{y}\n")
                    wiki_vocab.add(y)
    vocab_file.close()
