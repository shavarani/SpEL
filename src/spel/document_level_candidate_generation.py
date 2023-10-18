import torch
import torch.nn as nn
import torch.optim as optim
from wikipedia2vec import Wikipedia2Vec
import random
from tqdm import tqdm

from data_loader import tokenizer, ENWIKI20230827V2, AIDA20230827
from evaluate_local import SpELEvaluator
from configuration import get_checkpoints_dir, device


class LinearTransformation(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256):
        super(LinearTransformation, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class CandidateGenerator():
    def __init__(self) -> None:
        self.wiki2vec = None
        self.spel = None
        self.wiki2vec_entities = None
        self.wiki2vec_entities_list_for_negative_example_selection = None
        self.w_transform = None
        self.optimizer = None
        self.criterion = None
        self.load_wikipedia2vec()
        self.load_spel()
        self.init_nn()
    
    def load_wikipedia2vec(self):
        file_name = 'spel-wikipedia2vec-20230820.txt'
        print(f'downloading/loading {file_name} ...')
        torch.hub.download_url_to_file('https://vault.sfu.ca/index.php/s/cUDJU8DMwBag37u/download',
                                       get_checkpoints_dir() / file_name)
        self.wiki2vec = Wikipedia2Vec.load_text(get_checkpoints_dir() / file_name)
        self.wiki2vec_entities_list_for_negative_example_selection = [x.title for x in self.wiki2vec.dictionary.entities()]
        self.wiki2vec_entities = set(self.wiki2vec_entities_list_for_negative_example_selection)

    def get_negative_examples(self, count=1):
        if count > len(self.wiki2vec_entities_list_for_negative_example_selection):
            raise ValueError("\"count\" cannot be greater than the size of the indexed entities.")
        random_elements = random.sample(self.wiki2vec_entities_list_for_negative_example_selection, count)
        return random_elements
    
    def init_nn(self):
        self.w_transform = LinearTransformation(768, 768).to(device)
        self.optimizer = optim.Adam(self.w_transform.parameters(), lr=5e-4)
        self.criterion = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)


    def load_spel(self):
        self.spel = SpELEvaluator()
        self.spel.init_model_from_scratch(device=device)
        self.spel.shrink_classification_head_to_aida(device=device)
        self.spel.load_checkpoint(None, device=device, load_from_torch_hub=True, finetuned_after_step=3)

    def extract_data_loader_input_and_labels(self, item):
        tokens = item['tokens']
        mentions = item['mentions']
        mention_entity_probs = item['mention_entity_probs']
        sorted_mentions = sorted(list(zip(mentions, mention_entity_probs)), key=lambda x: x[1], reverse=True)
        annotations = set([x[0][0].replace('_', ' ') for x in sorted_mentions if x[0][0].replace('_', ' ') in self.wiki2vec_entities])
        sentence = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens))
        return sentence, annotations
    
    def encode_sentence(self, sentence):
        tokenized_mention = tokenizer([sentence])
        encoded_representations = self.spel.lm_module(torch.LongTensor(tokenized_mention.input_ids).to(device)).hidden_states
        cls_encoding = encoded_representations[-1][:, 0, :].detach() # of size 1 \times 768
        cls_encoding = self.w_transform(cls_encoding).squeeze(0)
        return cls_encoding

    def get_candidates(self, sentence, annotations):
        if not annotations:
            return []
        cls_encoding = self.encode_sentence(sentence)
        similars = self.wiki2vec.most_similar_by_vector(cls_encoding.detach().cpu().numpy(), count=len(annotations))
        return cls_encoding, [x[0].title for x in similars]
    
    def train(self, num_epochs=10, spl = 'train', save_model_name='w.pt', train_on_aida=True):
        for _ in range(num_epochs):
            if train_on_aida:
                print('training on AIDA20230827 ...')
                _iter_ = tqdm(AIDA20230827(split=spl, root=get_checkpoints_dir()))
            else:
                print('training on ENWIKI20230827V2 ...')
                _iter_ = tqdm(ENWIKI20230827V2(split=spl, root=get_checkpoints_dir()))
            total_loss = 0
            cnt_loss = 0
            for el in _iter_:
                s, anns = self.extract_data_loader_input_and_labels(el)
                if not anns: 
                    continue
                cls_encoding_transformed, hard_negatives = self.get_candidates(s, anns)
                hard_negatives = [x for x in hard_negatives if x not in anns] + \
                    [x for x in self.get_negative_examples(2 * len(anns) - len(hard_negatives)) if x not in anns and x not in hard_negatives]
                positive_examples = [torch.Tensor(self.wiki2vec.get_entity_vector(entity_name)).to(device) for entity_name in anns]
                negative_examples = [torch.Tensor(self.wiki2vec.get_entity_vector(entity_name)).to(device) for entity_name in hard_negatives]
                loss = 0.0
                cnt = 0.0
                for p in positive_examples:
                    for n in negative_examples:
                        loss += self.criterion(cls_encoding_transformed, p, n)
                        cnt += 1.0
                loss = loss / cnt
                total_loss += loss.detach().item()
                cnt_loss += 1.0
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                _iter_.set_description(f"Avg Loss: {total_loss/cnt_loss:.7f}")
            torch.save(self.w_transform.state_dict(), save_model_name)

    def inference(self, sentence, anns, load_model_name='w.pt'):
        self.w_transform.load_state_dict(torch.load(load_model_name))
        self.w_transform.eval()
        cls_encoding = self.encode_sentence(sentence)
        candidates = self.wiki2vec.most_similar_by_vector(cls_encoding.detach().cpu().numpy(), count=10)
        print(sentence)
        print("-" * 120)
        print(f"Reference candidates:\n>>{anns}")
        print("-" * 120)
        print(f"Predicted candidates:\n>>{candidates}")


if __name__ == "__main__":
    cgen = CandidateGenerator()
    cgen.train()
    sent = "CRICKET - LEICESTERSHIRE TAKE OVER AT TOP AFTER INNINGS VICTORY. LONDON 1996-08-30 West Indian all-rounder Phil Simmons took four for 38 on Friday as Leicestershire beat Somerset by an innings and 39 runs in two days to take over at the head of the county championship. Their stay on top, though, may be short-lived as title rivals Essex, Derbyshire and Surrey all closed in on victory while Kent made up for lost time in their rain-affected match against Nottinghamshire. After bowling Somerset out for 83 on the opening morning at Grace Road, Leicestershire extended their first innings by 94 runs before being bowled out for 296 with England discard Andy Caddick taking three for 83. Trailing by 213, Somerset got a solid start to their second innings before Simmons stepped in to bundle them out for 174. Essex, however, look certain to regain their top spot after Nasser Hussain and Peter Such gave them a firm grip on their match against Yorkshire at Headingley. Hussain, considered surplus to England's one-day requirements, struck 158, his first championship century of the season, as Essex reached 372 and took a first innings lead of 82. By the close Yorkshire had turned that into a 37-run advantage but off-spinner Such had scuttled their hopes, taking four for 24 in 48 balls and leaving them hanging on 119 for five and praying for rain. At the Oval, Surrey captain Chris Lewis, another man dumped by England, continued to silence his critics as he followed his four for 45 on Thursday with 80 not out on Friday in the match against Warwickshire. He was well backed by England hopeful Mark Butcher who made 70 as Surrey closed on 429 for seven, a lead of 234. Derbyshire kept up the hunt for their first championship title since 1936 by reducing Worcestershire to 133 for five in their second innings, still 100 runs away from avoiding an innings defeat. Australian Tom Moody took six for 82 but Chris Adams, 123, and Tim O'Gorman, 109, took Derbyshire to 471 and a first innings lead of 233. After the frustration of seeing the opening day of their match badly affected by the weather, Kent stepped up a gear to dismiss Nottinghamshire for 214. They were held up by a gritty 84 from Paul Johnson but ex-England fast bowler Martin McCague took four for 55. By stumps Kent had reached 108 for three."
    annos = [{"start": 10, "end": 24, "tag": "http://en.wikipedia.org/wiki/Leicestershire_County_Cricket_Club"}, {"start": 65, "end": 71, "tag": "http://en.wikipedia.org/wiki/London"}, {"start": 83, "end": 94, "tag": "http://en.wikipedia.org/wiki/West_Indies_cricket_team"}, {"start": 107, "end": 119, "tag": "http://en.wikipedia.org/wiki/Phil_Simmons"}, {"start": 150, "end": 164, "tag": "http://en.wikipedia.org/wiki/Leicestershire_County_Cricket_Club"}, {"start": 170, "end": 178, "tag": "http://en.wikipedia.org/wiki/Somerset_County_Cricket_Club"}, {"start": 332, "end": 337, "tag": "http://en.wikipedia.org/wiki/Essex_County_Cricket_Club"}, {"start": 339, "end": 349, "tag": "http://en.wikipedia.org/wiki/Derbyshire_County_Cricket_Club"}, {"start": 354, "end": 360, "tag": "http://en.wikipedia.org/wiki/Surrey_County_Cricket_Club"}, {"start": 392, "end": 396, "tag": "http://en.wikipedia.org/wiki/Kent_County_Cricket_Club"}, {"start": 456, "end": 471, "tag": "http://en.wikipedia.org/wiki/Nottinghamshire_County_Cricket_Club"}, {"start": 487, "end": 495, "tag": "http://en.wikipedia.org/wiki/Somerset_County_Cricket_Club"}, {"start": 533, "end": 543, "tag": "http://en.wikipedia.org/wiki/Grace_Road"}, {"start": 545, "end": 559, "tag": "http://en.wikipedia.org/wiki/Leicestershire_County_Cricket_Club"}, {"start": 637, "end": 644, "tag": "http://en.wikipedia.org/wiki/England_cricket_team"}, {"start": 653, "end": 665, "tag": "http://en.wikipedia.org/wiki/Andrew_Caddick"}, {"start": 704, "end": 712, "tag": "http://en.wikipedia.org/wiki/Somerset_County_Cricket_Club"}, {"start": 762, "end": 769, "tag": "http://en.wikipedia.org/wiki/Phil_Simmons"}, {"start": 809, "end": 814, "tag": "http://en.wikipedia.org/wiki/Essex_County_Cricket_Club"}, {"start": 869, "end": 883, "tag": "http://en.wikipedia.org/wiki/Nasser_Hussain"}, {"start": 888, "end": 898, "tag": "http://en.wikipedia.org/wiki/Peter_Such"}, {"start": 944, "end": 953, "tag": "http://en.wikipedia.org/wiki/Yorkshire_County_Cricket_Club"}, {"start": 957, "end": 967, "tag": "http://en.wikipedia.org/wiki/Headingley"}, {"start": 969, "end": 976, "tag": "http://en.wikipedia.org/wiki/Nasser_Hussain"}, {"start": 1000, "end": 1007, "tag": "http://en.wikipedia.org/wiki/England_cricket_team"}, {"start": 1093, "end": 1098, "tag": "http://en.wikipedia.org/wiki/Essex_County_Cricket_Club"}, {"start": 1161, "end": 1170, "tag": "http://en.wikipedia.org/wiki/Yorkshire_County_Cricket_Club"}, {"start": 1227, "end": 1231, "tag": "http://en.wikipedia.org/wiki/Peter_Such"}, {"start": 1359, "end": 1363, "tag": "http://en.wikipedia.org/wiki/The_Oval"}, {"start": 1365, "end": 1371, "tag": "http://en.wikipedia.org/wiki/Surrey_County_Cricket_Club"}, {"start": 1380, "end": 1391, "tag": "http://en.wikipedia.org/wiki/Chris_Lewis_(cricketer)"}, {"start": 1415, "end": 1422, "tag": "http://en.wikipedia.org/wiki/England_cricket_team"}, {"start": 1547, "end": 1559, "tag": "http://en.wikipedia.org/wiki/Warwickshire_County_Cricket_Club"}, {"start": 1583, "end": 1590, "tag": "http://en.wikipedia.org/wiki/England_cricket_team"}, {"start": 1599, "end": 1611, "tag": "http://en.wikipedia.org/wiki/Mark_Butcher"}, {"start": 1627, "end": 1633, "tag": "http://en.wikipedia.org/wiki/Surrey_County_Cricket_Club"}, {"start": 1674, "end": 1684, "tag": "http://en.wikipedia.org/wiki/Derbyshire_County_Cricket_Club"}, {"start": 1760, "end": 1774, "tag": "http://en.wikipedia.org/wiki/Worcestershire_County_Cricket_Club"}, {"start": 1869, "end": 1879, "tag": "http://en.wikipedia.org/wiki/Australia"}, {"start": 1880, "end": 1889, "tag": "http://en.wikipedia.org/wiki/Tom_Moody"}, {"start": 1910, "end": 1921, "tag": "http://en.wikipedia.org/wiki/Chris_Adams_(cricketer)"}, {"start": 1956, "end": 1966, "tag": "http://en.wikipedia.org/wiki/Derbyshire_County_Cricket_Club"}, {"start": 2101, "end": 2105, "tag": "http://en.wikipedia.org/wiki/Kent_County_Cricket_Club"}, {"start": 2135, "end": 2150, "tag": "http://en.wikipedia.org/wiki/Nottinghamshire_County_Cricket_Club"}, {"start": 2198, "end": 2210, "tag": "http://en.wikipedia.org/wiki/Paul_Johnson_(cricketer)"}, {"start": 2238, "end": 2252, "tag": "http://en.wikipedia.org/wiki/Martin_McCague"}, {"start": 2281, "end": 2285, "tag": "http://en.wikipedia.org/wiki/Kent_County_Cricket_Club"}]
    annos = [x["tag"] for x in annos]
    cgen.inference(sent, annos)
