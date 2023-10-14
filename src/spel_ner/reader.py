"""
This script provides different data readers to the training/evaluation scripts in SpEL-NER package.
"""
from datasets import load_dataset
from transformers import AutoTokenizer

BERT_MODEL_NAME = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME, add_prefix_space=True)

class Dataset:
    def __init__(self):
        self.label2id = {}
        self.id2label = []
        self.dataset = []

    @property
    def size(self):
        return len(self.id2label)

class CoNLL2003(Dataset):
    def __init__(self):
        super().__init__()
        DATASET_NAME = "tner/conll2003"
        self.label2id = {
            "O": 0, "B-ORG": 1, "B-MISC": 2, "B-PER": 3, "I-PER": 4, "B-LOC": 5, "I-ORG": 6, "I-MISC": 7, "I-LOC": 8}
        self.id2label = [k for k, v in sorted(self.label2id.items(), key=lambda x: x[1])]
        def preprocess_function(examples):
            tokens = tokenizer(examples["tokens"], is_split_into_words=True, padding="max_length", truncation=True)
            labels = []
            for tags, alignments in zip(examples["tags"], [x.word_ids for x in tokens.encodings]):
                label_ids = []
                for a_ind, a in enumerate(alignments):
                    label_ids.append(
                        self.id2label.index('O') if a is None else self.id2label.index(self.id2label[tags[a]]))
                labels.append(label_ids)
            return {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"], "labels": labels}
        self.original_dataset = load_dataset(DATASET_NAME)
        self.preprocessed_dataset = self.original_dataset.map(
            preprocess_function, batched=True, remove_columns=['tokens', 'tags'])