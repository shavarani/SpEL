"""
The data loader to serve the global candidate generation training process.
"""
import random
from pynif import NIFCollection
from torch.utils.data import DataLoader
import torch
from transformers import BatchEncoding
from wikipedia2vec import Wikipedia2Vec

from spel.data_loader import tokenizer, ENWIKI20230827V2, ENWIKI20230827V2Config, AIDA20230827, AIDA20230827Config
from spel.evaluate_local import SpELEvaluator
from spel.configuration import get_checkpoints_dir, device

def load_spel():
    print('Loading fine-tuned SpEL model ...')
    spel = SpELEvaluator()
    spel.init_model_from_scratch(device=device)
    spel.shrink_classification_head_to_aida(device=device)
    spel.load_checkpoint(None, device=device, load_from_torch_hub=True, finetuned_after_step=3)
    return spel

class W2V:
    def __init__(self):
        self.wiki2vec = None
        self.wiki2vec_entities = None
        self.wiki2vec_entities_list_for_negative_example_selection = None
        self.load_wikipedia2vec()

    def get_negative_examples(self, count=1):
        if count > len(self.wiki2vec_entities_list_for_negative_example_selection):
            raise ValueError("\"count\" cannot be greater than the size of the indexed entities.")
        random_elements = random.sample(self.wiki2vec_entities_list_for_negative_example_selection, count)
        return random_elements

    def load_wikipedia2vec(self):
        print('Loading Wikipedia2Vec entities ...')
        file_name = 'spel-wikipedia2vec-20230820.txt'
        if not (get_checkpoints_dir() / file_name).exists():
            print(f'downloading {file_name} ...')
            torch.hub.download_url_to_file('https://vault.sfu.ca/index.php/s/cUDJU8DMwBag37u/download',
                                           get_checkpoints_dir() / file_name)
        self.wiki2vec = Wikipedia2Vec.load_text(get_checkpoints_dir() / file_name)
        self.wiki2vec_entities_list_for_negative_example_selection = [x.title
                                                                      for x in self.wiki2vec.dictionary.entities()]
        self.wiki2vec_entities = set(self.wiki2vec_entities_list_for_negative_example_selection)

SpEL_instance = load_spel()
wikipedia2vec_instance = W2V()

def spel_annotate(sentence):
    collection = NIFCollection(uri="http://spel.sfu.ca")
    context = collection.add_context(uri="http://spel.sfu.ca/doc1", mention=sentence)
    SpEL_instance.annotate(context)
    annotations = [{"start": phrase.beginIndex, "end": phrase.endIndex, "tag": phrase.taIdentRef}
                   for phrase in collection.contexts[0]._context.phrases]
    return annotations


class ELDataset(torch.utils.data.Dataset):
    def __init__(self, train_on_aida=True, split='train'):
        if train_on_aida:
            print('Loading AIDA20230827 ...')
            self.data_iterator = iter(AIDA20230827(split=split, root=get_checkpoints_dir()))
            self.dataset_len = AIDA20230827Config.NUM_LINES[split]
        else:
            print('Loading ENWIKI20230827V2 ...')
            self.data_iterator = iter(ENWIKI20230827V2(split=split, root=get_checkpoints_dir()))
            self.dataset_len = ENWIKI20230827V2Config.NUM_LINES[split]
        self.tokenizer = tokenizer

    def __len__(self):
        return self.dataset_len

    @staticmethod
    def get_mention(data, index):
        mentions = data['mentions'][index]
        mention_entity_probs = data['mention_entity_probs'][index]
        return sorted(list(zip(mentions, mention_entity_probs)),
                      key=lambda x: x[1], reverse=True)[0][0].replace('_', ' ')

    def __getitem__(self, idx):
        data = next(self.data_iterator)
        sentence = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(data['tokens']))
        spans = []
        spans.append({'start': 0, 'end': 1, 'mention': self.get_mention(data, 0), 'subwords': [data['tokens'][0]]})
        for i in range(1, len(data['mentions'])):
            mention = self.get_mention(data, i)
            if spans[-1]['mention'] == mention:
                spans[-1]['end'] = i + 1
                spans[-1]['subwords'].append(data['tokens'][i])
            else:
                spans.append({'start': i, 'end': i + 1, 'mention': mention, 'subwords': [data['tokens'][i]]})
        encoding = self.tokenizer.convert_tokens_to_ids(data['tokens'])
        spans = [x for x in spans if x['mention'] != '|||O|||']
        return {'sentence': sentence, 'tokens': data['tokens'], 'spans': spans, 'encoding': encoding}

class ELDataLoader(DataLoader):
    """
    Can be loaded using ELDataLoader(ELDataset(train_on_aida=True/False), batch_size=batch_size)
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, flatten_spans=True):
        super().__init__(dataset, batch_size, shuffle, collate_fn=self.collate_fn)
        self.flatten_spans = flatten_spans

    def collate_fn(self, batch):
        all_lens = [len(item["encoding"]) for item in batch]
        maxlen = max(all_lens)
        encodings = torch.LongTensor(
            [item['encoding'] + [0] * (maxlen - len(item['encoding'])) for item in batch]).to(device)
        # For now only keeping the last layer, however, we could keep all the layers or average them out!
        enc = SpEL_instance.lm_module(encodings).hidden_states[-1].detach()
        embeddings = [enc[s_i][:s_len] for s_i, s_len in enumerate(all_lens)]
        cls_repr = [x[0].cpu() for x in embeddings]
        sentences = [item['sentence'] for item in batch]
        spans = [[
                    {
                        'embeddings': {
                            'start': embeddings[item_index][sp['start']].cpu(),
                            'end': embeddings[item_index][sp['end'] - 1].cpu(),
                            'average': (embeddings[item_index][sp['start']:sp['end']].sum(
                                dim=0) / (sp['end'] - sp['start'])).cpu(),
                        },
                        'mention': sp['mention']
                    } for sp in item['spans']
                ] for item_index, item in enumerate(batch)]
        return BatchEncoding({
            'sentences': sentences,
            'spans': [x for y in spans for x in y] if self.flatten_spans else spans,
            'cls_representations': cls_repr
        })

class TripleVectorDataProvider:
    def __init__(self, train_on_aida, sentence_batch_size, expected_negative_examples, default_key = 'start'):
        self.dataset = ELDataLoader(ELDataset(train_on_aida=train_on_aida), batch_size=sentence_batch_size)
        self.default_key = default_key
        self.expected_negative_examples = expected_negative_examples

    def __len__(self):
        return self.dataset.dataset.dataset_len

    def instances(self):
        for batch in self.dataset:
            for span in batch.spans:
                mention = span['mention']
                if mention not in wikipedia2vec_instance.wiki2vec_entities:
                    continue
                lm_embedding = span['embeddings'][self.default_key]
                mention_vector = wikipedia2vec_instance.wiki2vec.get_entity_vector(mention)
                for hard_negative in wikipedia2vec_instance.wiki2vec.most_similar_by_vector(
                        mention_vector, self.expected_negative_examples):
                    if hard_negative[0].title == mention:
                        continue
                    hard_negative_vector = wikipedia2vec_instance.wiki2vec.get_entity_vector(hard_negative[0].title)
                    yield lm_embedding, mention_vector, hard_negative_vector

class TripleVectorDataset(torch.utils.data.Dataset):
    def __init__(self, train_on_aida, batch_size, expected_negative_examples, default_key = 'start'):
        self.data_generator = TripleVectorDataProvider(
            train_on_aida, batch_size, expected_negative_examples, default_key).instances()
        self.train_on_aida = train_on_aida
    
    def __len__(self):
        return 200000 if self.train_on_aida else 300000000

    def __getitem__(self, idx):
        sample = next(self.data_generator)
        return sample[0].to(device), torch.FloatTensor(sample[1]).to(device), torch.FloatTensor(sample[2]).to(device)
    
def get_dataset(train_on_aida, triple_vector_batch_size, expected_negative_examples, default_key = 'start',
                lm_batch_size=10):
    return DataLoader(TripleVectorDataset(train_on_aida, lm_batch_size, expected_negative_examples, default_key),  
                      batch_size=triple_vector_batch_size)
