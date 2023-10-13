import torch.nn as nn
from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput


class NERModel(PreTrainedModel):
    def __init__(self, config, num_labels, freeze_bert=False):
        super(NERModel, self).__init__(config)
        self.num_labels = num_labels
        self.model = AutoModel.from_config(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()
        if freeze_bert:
            for param in self.model.parameters():
                param.requires_grad = False
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of Trainable Parameters: {trainable_params}")

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                             position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)
        loss = 0.0

        if labels is not None:
            active_loss_ner = attention_mask.view(-1) == 1
            active_logits_ner = logits.view(-1, self.num_labels)[active_loss_ner]
            active_labels_ner = labels.view(-1)[active_loss_ner]
            loss = self.loss_fn(active_logits_ner, active_labels_ner)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )