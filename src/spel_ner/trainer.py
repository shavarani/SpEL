"""
The training script to train and evaluate the SpEL-NER model.
"""
import torch
import numpy as np
from transformers import AutoConfig, TrainingArguments, Trainer
from sklearn.metrics import f1_score
from tqdm import tqdm

from spel_ner.reader import tokenizer, CoNLL2003, BERT_MODEL_NAME
from spel_ner.model import NERModel
from spel_ner.evaluate import span_f1

dataset_object = CoNLL2003()

def compute_metrics(p):
    labels = p.label_ids
    preds = p.predictions.argmax(-1)
    f1_scores = []
    for label_id in range(dataset_object.size):
        label_preds = [pred[label_id] for pred in preds]
        label_true = [true[label_id] for true in labels]
        label_f1 = f1_score(label_true, label_preds, average='micro')
        f1_scores.append(label_f1)
    avg_f1 = np.mean(f1_scores)
    return {"f1": avg_f1}

def train_model(save_model_name="spel-ner-model"):
    model = NERModel(AutoConfig.from_pretrained(BERT_MODEL_NAME), num_labels=dataset_object.size)

    training_args = TrainingArguments(
        output_dir=f"./{save_model_name}",
        num_train_epochs=30,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        save_steps=5000,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=5000,
        logging_dir="./logs",
        learning_rate=8e-5,
        remove_unused_columns=False,
        fp16=True,
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=None,
        compute_metrics=compute_metrics,
        train_dataset=dataset_object.preprocessed_dataset["train"],
        eval_dataset=dataset_object.preprocessed_dataset["validation"],
    )

    trainer.train()
    model.save_pretrained(save_model_name)

def resolve_cluster_predictions(word_cluster):
    #TODO find a better resolution strategy!
    return word_cluster[0][1]

def evaluate(load_model_name="spel-ner-model"):
    gold_label_dict = [k for k, v in sorted(dataset_object.label2id.items(), key=lambda x: x[1])]
    cfg = AutoConfig.from_pretrained(load_model_name)
    model = NERModel(cfg, num_labels=dataset_object.size)
    model.load_state_dict(torch.load(f"{load_model_name}/pytorch_model.bin"))
    pred_list = []
    label_list = []
    model.eval()
    for elem in tqdm(dataset_object.original_dataset['validation']):
        tokens = elem['tokens']
        inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt")
        outputs = model(**inputs)
        predicted_labels = outputs.logits.argmax(dim=2)
        subword_predictions = [dataset_object.id2label[label_id] for label_id in predicted_labels[0]]
        merged_predictions = [(subword, subword_prediction, word_id) for subword_prediction, subword, word_id in zip(
            subword_predictions, inputs.encodings[0].tokens, inputs.encodings[0].word_ids)
                              if subword not in ['<s>', '</s>']]
        word_content = []
        for subword_content in merged_predictions:
            if not word_content or subword_content[-1] != word_content[-1][-1][-1]:
                word_content.append([])
            word_content[-1].append(subword_content)
        predictions = []
        for word_cluster in word_content:
            if len(word_cluster) == 1:
                word_cluster = word_cluster[0]
                predictions.append(word_cluster[1])
            else:
                p = resolve_cluster_predictions(word_cluster)
                # t = tokenizer.convert_tokens_to_string([x[0]for x in word_cluster])
                predictions.append(p)
        pred_list.append(predictions)
        label_list.append([gold_label_dict[x] for x in elem['tags']])
    eval_results = span_f1(pred_list, label_list)
    print(eval_results)

if __name__ == '__main__':
    train_model()
    evaluate()