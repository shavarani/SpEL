import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NERModel(PreTrainedModel):
    """
    CoNLL2003 (t-ner version) scores considering RoBERTa-base as the encoder:
        'parameter_size': 124,652,553
        'micro/f1': 0.8024, 'micro/recall': 0.8125, 'micro/precision': 0.7926,
        'macro/f1': 0.7983, 'macro/recall': 0.8049, 'macro/precision': 0.7926,
        'per_entity_metric': {
            'location':     {'f1': 0.8877, 'precision': 0.8882, 'recall': 0.8873},
            'organization': {'f1': 0.7355, 'precision': 0.7161, 'recall': 0.7561},
            'other':        {'f1': 0.8006, 'precision': 0.8202, 'recall': 0.7819},
            'person':       {'f1': 0.7693, 'precision': 0.7460, 'recall': 0.7942}
        }

    """
    def __init__(self, config, num_labels, freeze_bert=False):
        super(NERModel, self).__init__(config)
        self.num_labels = num_labels
        self.model = AutoModel.from_config(config)
        self.load_encoder_model()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()
        if freeze_bert:
            for param in self.model.parameters():
                param.requires_grad = False
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of Trainable Parameters: {trainable_params}")

    def load_encoder_model(self):
        return


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

class SpELNERModel(NERModel):
    def load_encoder_model(self):
        file_name = "spel-base-step-3.pt"
        checkpoint = torch.hub.load_state_dict_from_url('https://vault.sfu.ca/index.php/s/HpQ3PMm6A3y1NBl/download',
                                                        model_dir=".checkpoints", map_location="cpu",
                                                        file_name=file_name)
        self.model.load_state_dict(checkpoint["bert_lm"], strict=False)
        self.model.to(device)
        print(f"Loaded {file_name}")