import torch
from torch import nn
from transformers import AutoModelForTokenClassification


def loss_fn(output, target, mask, num_labels):
    lfn = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
        active_loss,
        target.view(-1),
        torch.tensor(lfn.ignore_index).type_as(target)
    )
    loss = lfn(active_logits, active_labels)
    return loss


class NERModel(nn.Module):
    def __init__(self, model_name, num_labels, cache_dir, is_freeze=False):
        super(NERModel, self).__init__()
        self.bert = AutoModelForTokenClassification.from_pretrained(model_name,
                                                                    output_hidden_states=True,
                                                                    cache_dir=cache_dir)  # 默认 return_dict=True
        self.num_labels = num_labels
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1024 * 4, self.num_labels, bias=False),
        )

        if is_freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        hidden_states = torch.cat(tuple([outputs.hidden_states[i] for i in [-1, -2, -3, -4]]),
                                  dim=-1)  # [bs, seq_len, hidden_dim*4-model]
        first_hidden_states = hidden_states[:, 0, :]  # [bs, hidden_dim*4-model]
        logits = self.fc(hidden_states)
        if labels is None:
            return logits, None
        else:
            loss = loss_fn(logits, labels, attention_mask, self.num_labels)
            return logits, loss
