import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from spel.configuration import device, get_base_model_name
from cgen.data_loader import get_dataset, spel_annotate, tokenizer, SpEL_instance, wikipedia2vec_instance

class Transformation(nn.Module):
    def __init__(self, spel_lm_size, w2v_size=768, hidden_size=256):
        super(Transformation, self).__init__()
        self.w2v_map = nn.Linear(w2v_size, hidden_size)
        self.spel_map = nn.Linear(spel_lm_size, hidden_size)
        nn.init.xavier_uniform_(self.w2v_map.weight)
        nn.init.xavier_uniform_(self.spel_map.weight)


class SpELTransformation(nn.Module):
    def __init__(self, spel_lm_size, w2v_size, hidden_size=256):
        super(SpELTransformation, self).__init__()
        self.fc1 = nn.Linear(spel_lm_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, w2v_size)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class Generator:
    def __init__(self, only_transform_spel=True):
        self.only_transform_spel = only_transform_spel
        transform_class = SpELTransformation if only_transform_spel else Transformation
        self.transformation = transform_class(1024 if get_base_model_name() == 'roberta-large' else 768, 768).to(device)

    def train(self, train_on_aida=True, lm_batch_size=1, triple_vector_batch_size=64, lr=5e-4, num_epochs=1, checkpoint_every=1000,
              save_model_name='generator_w.pt', expected_negative_examples=10, default_key = 'start'):
        optimizer = optim.Adam(self.transformation.parameters(), lr=lr)
        criterion = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
        total_loss = 0
        cnt_loss = 0
        for _ in range(num_epochs):
            # Run the training code to train self.transformation
            dataset = tqdm(get_dataset(train_on_aida, triple_vector_batch_size, expected_negative_examples, default_key = default_key, lm_batch_size=lm_batch_size))
            for batch_id, batch in enumerate(dataset):
                if batch_id > 0 and batch_id % checkpoint_every == 0:
                    print('checkpointing ...')
                    torch.save(self.transformation.state_dict(), save_model_name)
                anchor, positive, negative = batch
                if self.only_transform_spel:
                    anchor = self.transformation(anchor)
                else:
                    anchor = self.transformation.spel_map(anchor)
                    positive = self.transformation.w2v_map(positive)
                    negative = self.transformation.w2v_map(negative)
                loss = criterion(anchor, positive, negative)
                del batch[:]
                total_loss += loss.detach().item()
                cnt_loss += 1.0
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                dataset.set_description(f"Avg Loss: {total_loss/cnt_loss:.7f}")
            torch.save(self.transformation.state_dict(), save_model_name)

    def transform_lm_embeddings_to_w2vec_space(self, e):
        return self.transformation(e) if self.only_transform_spel else (
            self.transformation.spel_map(e).matmul(self.transformation.w2v_map.weight))

    def inference(self, sentence, anns, load_model_name='generator_w.pt', c_count=3):
        self.transformation.load_state_dict(torch.load(load_model_name))
        self.transformation.eval()
        el_annotations = spel_annotate(sentence)
        tokenized_mention = tokenizer([sentence])
        char_to_subword_mapping = {}
        offsets = []
        # correcting the offsets so that all the characters can be assigned a valid subword id
        for (char_start, char_end) in tokenized_mention.encodings[0].offsets[1:-1]:
            if offsets and offsets[-1][1] != char_start:
                offsets[-1] = (offsets[-1][0], char_start)
            offsets.append((char_start, char_end))
        for subword_id, (char_start, char_end) in enumerate(offsets):
            for c_id in range(char_start, char_end):
                char_to_subword_mapping[c_id] = subword_id
        subword_preds = [{'start_subword': char_to_subword_mapping[x['start']], 'end_subword': char_to_subword_mapping[x['end']], **x} for x in el_annotations]
        # subword_actuals = [{'start': char_to_subword_mapping[x['start']], 'end': char_to_subword_mapping[x['end']], 'tag': x['tag']} for x in anns]
        embeddings = SpEL_instance.lm_module(torch.LongTensor(tokenized_mention.input_ids).to(device)).hidden_states[-1][0]
        embeddings = self.transform_lm_embeddings_to_w2vec_space(embeddings)
        # TODO once decided which one holds better results, reduce the most_similar_by_vector to only one of start, end, and average!
        span_predictions = [
            {
                #'embeddings': {
                #    'start': embeddings[sp['start_subword']].cpu().numpy(),
                #    'end': embeddings[sp['end_subword']].cpu().numpy(),
                #    'average': (embeddings[sp['start_subword']:sp['end_subword'] + 1].sum(
                #        dim=0) / (sp['end_subword'] + 1 - sp['start_subword'])).cpu().numpy(),
                #},
                'candidates': {
                    'start': wikipedia2vec_instance.wiki2vec.most_similar_by_vector(embeddings[sp['start_subword']].detach().cpu().numpy(), count=c_count),
                    'end': wikipedia2vec_instance.wiki2vec.most_similar_by_vector(embeddings[sp['end_subword']].detach().cpu().numpy(), count=c_count),
                    'average': wikipedia2vec_instance.wiki2vec.most_similar_by_vector((embeddings[sp['start_subword']:sp['end_subword'] + 1].sum(
                        dim=0) / (sp['end_subword'] + 1 - sp['start_subword'])).detach().cpu().numpy(), count=c_count),
                },
                **sp
            } for sp in subword_preds
        ]
        print(sentence)
        print("-" * 120)
        print(f"Reference candidates:\n>>{anns}")
        print("-" * 120)
        print(f"Predicted candidates:\n>>{span_predictions}")

if __name__ == "__main__":
    cgen = Generator()
    cgen.train(train_on_aida=True, lm_batch_size=64, triple_vector_batch_size=768, lr=5e-4, num_epochs=50, checkpoint_every=2000,
              save_model_name='generator_w.pt', expected_negative_examples=10, default_key = 'start')
    sent = "CRICKET - LEICESTERSHIRE TAKE OVER AT TOP AFTER INNINGS VICTORY. LONDON 1996-08-30 West Indian all-rounder Phil Simmons took four for 38 on Friday as Leicestershire beat Somerset by an innings and 39 runs in two days to take over at the head of the county championship. Their stay on top, though, may be short-lived as title rivals Essex, Derbyshire and Surrey all closed in on victory while Kent made up for lost time in their rain-affected match against Nottinghamshire. After bowling Somerset out for 83 on the opening morning at Grace Road, Leicestershire extended their first innings by 94 runs before being bowled out for 296 with England discard Andy Caddick taking three for 83. Trailing by 213, Somerset got a solid start to their second innings before Simmons stepped in to bundle them out for 174. Essex, however, look certain to regain their top spot after Nasser Hussain and Peter Such gave them a firm grip on their match against Yorkshire at Headingley. Hussain, considered surplus to England's one-day requirements, struck 158, his first championship century of the season, as Essex reached 372 and took a first innings lead of 82. By the close Yorkshire had turned that into a 37-run advantage but off-spinner Such had scuttled their hopes, taking four for 24 in 48 balls and leaving them hanging on 119 for five and praying for rain. At the Oval, Surrey captain Chris Lewis, another man dumped by England, continued to silence his critics as he followed his four for 45 on Thursday with 80 not out on Friday in the match against Warwickshire. He was well backed by England hopeful Mark Butcher who made 70 as Surrey closed on 429 for seven, a lead of 234. Derbyshire kept up the hunt for their first championship title since 1936 by reducing Worcestershire to 133 for five in their second innings, still 100 runs away from avoiding an innings defeat. Australian Tom Moody took six for 82 but Chris Adams, 123, and Tim O'Gorman, 109, took Derbyshire to 471 and a first innings lead of 233. After the frustration of seeing the opening day of their match badly affected by the weather, Kent stepped up a gear to dismiss Nottinghamshire for 214. They were held up by a gritty 84 from Paul Johnson but ex-England fast bowler Martin McCague took four for 55. By stumps Kent had reached 108 for three."
    annos = [{"start": 10, "end": 24, "tag": "http://en.wikipedia.org/wiki/Leicestershire_County_Cricket_Club"}, {"start": 65, "end": 71, "tag": "http://en.wikipedia.org/wiki/London"}, {"start": 83, "end": 94, "tag": "http://en.wikipedia.org/wiki/West_Indies_cricket_team"}, {"start": 107, "end": 119, "tag": "http://en.wikipedia.org/wiki/Phil_Simmons"}, {"start": 150, "end": 164, "tag": "http://en.wikipedia.org/wiki/Leicestershire_County_Cricket_Club"}, {"start": 170, "end": 178, "tag": "http://en.wikipedia.org/wiki/Somerset_County_Cricket_Club"}, {"start": 332, "end": 337, "tag": "http://en.wikipedia.org/wiki/Essex_County_Cricket_Club"}, {"start": 339, "end": 349, "tag": "http://en.wikipedia.org/wiki/Derbyshire_County_Cricket_Club"}, {"start": 354, "end": 360, "tag": "http://en.wikipedia.org/wiki/Surrey_County_Cricket_Club"}, {"start": 392, "end": 396, "tag": "http://en.wikipedia.org/wiki/Kent_County_Cricket_Club"}, {"start": 456, "end": 471, "tag": "http://en.wikipedia.org/wiki/Nottinghamshire_County_Cricket_Club"}, {"start": 487, "end": 495, "tag": "http://en.wikipedia.org/wiki/Somerset_County_Cricket_Club"}, {"start": 533, "end": 543, "tag": "http://en.wikipedia.org/wiki/Grace_Road"}, {"start": 545, "end": 559, "tag": "http://en.wikipedia.org/wiki/Leicestershire_County_Cricket_Club"}, {"start": 637, "end": 644, "tag": "http://en.wikipedia.org/wiki/England_cricket_team"}, {"start": 653, "end": 665, "tag": "http://en.wikipedia.org/wiki/Andrew_Caddick"}, {"start": 704, "end": 712, "tag": "http://en.wikipedia.org/wiki/Somerset_County_Cricket_Club"}, {"start": 762, "end": 769, "tag": "http://en.wikipedia.org/wiki/Phil_Simmons"}, {"start": 809, "end": 814, "tag": "http://en.wikipedia.org/wiki/Essex_County_Cricket_Club"}, {"start": 869, "end": 883, "tag": "http://en.wikipedia.org/wiki/Nasser_Hussain"}, {"start": 888, "end": 898, "tag": "http://en.wikipedia.org/wiki/Peter_Such"}, {"start": 944, "end": 953, "tag": "http://en.wikipedia.org/wiki/Yorkshire_County_Cricket_Club"}, {"start": 957, "end": 967, "tag": "http://en.wikipedia.org/wiki/Headingley"}, {"start": 969, "end": 976, "tag": "http://en.wikipedia.org/wiki/Nasser_Hussain"}, {"start": 1000, "end": 1007, "tag": "http://en.wikipedia.org/wiki/England_cricket_team"}, {"start": 1093, "end": 1098, "tag": "http://en.wikipedia.org/wiki/Essex_County_Cricket_Club"}, {"start": 1161, "end": 1170, "tag": "http://en.wikipedia.org/wiki/Yorkshire_County_Cricket_Club"}, {"start": 1227, "end": 1231, "tag": "http://en.wikipedia.org/wiki/Peter_Such"}, {"start": 1359, "end": 1363, "tag": "http://en.wikipedia.org/wiki/The_Oval"}, {"start": 1365, "end": 1371, "tag": "http://en.wikipedia.org/wiki/Surrey_County_Cricket_Club"}, {"start": 1380, "end": 1391, "tag": "http://en.wikipedia.org/wiki/Chris_Lewis_(cricketer)"}, {"start": 1415, "end": 1422, "tag": "http://en.wikipedia.org/wiki/England_cricket_team"}, {"start": 1547, "end": 1559, "tag": "http://en.wikipedia.org/wiki/Warwickshire_County_Cricket_Club"}, {"start": 1583, "end": 1590, "tag": "http://en.wikipedia.org/wiki/England_cricket_team"}, {"start": 1599, "end": 1611, "tag": "http://en.wikipedia.org/wiki/Mark_Butcher"}, {"start": 1627, "end": 1633, "tag": "http://en.wikipedia.org/wiki/Surrey_County_Cricket_Club"}, {"start": 1674, "end": 1684, "tag": "http://en.wikipedia.org/wiki/Derbyshire_County_Cricket_Club"}, {"start": 1760, "end": 1774, "tag": "http://en.wikipedia.org/wiki/Worcestershire_County_Cricket_Club"}, {"start": 1869, "end": 1879, "tag": "http://en.wikipedia.org/wiki/Australia"}, {"start": 1880, "end": 1889, "tag": "http://en.wikipedia.org/wiki/Tom_Moody"}, {"start": 1910, "end": 1921, "tag": "http://en.wikipedia.org/wiki/Chris_Adams_(cricketer)"}, {"start": 1956, "end": 1966, "tag": "http://en.wikipedia.org/wiki/Derbyshire_County_Cricket_Club"}, {"start": 2101, "end": 2105, "tag": "http://en.wikipedia.org/wiki/Kent_County_Cricket_Club"}, {"start": 2135, "end": 2150, "tag": "http://en.wikipedia.org/wiki/Nottinghamshire_County_Cricket_Club"}, {"start": 2198, "end": 2210, "tag": "http://en.wikipedia.org/wiki/Paul_Johnson_(cricketer)"}, {"start": 2238, "end": 2252, "tag": "http://en.wikipedia.org/wiki/Martin_McCague"}, {"start": 2281, "end": 2285, "tag": "http://en.wikipedia.org/wiki/Kent_County_Cricket_Club"}]
    cgen.inference(sent, annos)