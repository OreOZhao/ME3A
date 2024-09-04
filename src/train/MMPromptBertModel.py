from transformers import BertModel
import torch as t
from config.KBConfig import *
from tools.Announce import Announce
import torch.nn as nn


class MMPromptBertModel(t.nn.Module):
    def __init__(self, pretrain_bert_path, prompt_labels_tids, vis_token, prefix=2):
        super(MMPromptBertModel, self).__init__()
        self.bert_config = BertConfig.from_pretrained(pretrain_bert_path)
        self.num_labels = num_prompt_labels
        self.label_list = prompt_labels_tids
        self.bert_model = BertModel.from_pretrained(pretrain_bert_path, config=self.bert_config)
        self.bert_model.resize_token_embeddings(len_tokenizer)
        self.out_linear_layer = t.nn.Linear(self.bert_config.hidden_size, bert_output_dim)
        self.prompt_linear_layer = t.nn.Linear(self.bert_config.hidden_size, self.bert_config.vocab_size)
        self.prompt_logits_linear_layer = t.nn.Linear(self.bert_config.hidden_size, 2)
        self.dropout = t.nn.Dropout(p=0.1)
        self.prefix = prefix
        self.vis_token = vis_token
        self.vision_mapping = nn.Sequential(
            nn.Linear(visual_hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.bert_config.hidden_size * self.prefix),
            nn.Sigmoid()
        )
        print(Announce.printMessage(), '--------Init MMPromptBertModel--------')

    def get_cls_output(self, tids, mask_cls, vis):
        bs = tids.shape[0]
        inputs_embeds = self.bert_model.embeddings(input_ids=tids)
        vis_prefix = self.vision_mapping(vis).reshape(-1, self.bert_config.hidden_size)
        vis_idx_0 = (tids == self.vis_token).nonzero(as_tuple=True)[0]
        vis_idx_1 = (tids == self.vis_token).nonzero(as_tuple=True)[1]
        inputs_embeds[vis_idx_0, vis_idx_1] = vis_prefix
        bert_out = self.bert_model(inputs_embeds=inputs_embeds, attention_mask=mask_cls)
        last_hidden_state = bert_out.last_hidden_state
        cls = last_hidden_state[:, 0]
        output = self.dropout(cls)
        output = self.out_linear_layer(output)
        return output

    def _get_cls_output(self, inputs_embeds, mask_cls):
        bert_out = self.bert_model(inputs_embeds=inputs_embeds, attention_mask=mask_cls)
        last_hidden_state = bert_out.last_hidden_state
        cls = last_hidden_state[:, 0]
        output = self.dropout(cls)
        output = self.out_linear_layer(output)
        return output

    def _get_cls2_output(self, inputs_embeds, mask_cls):
        bert_out = self.bert_model(inputs_embeds=inputs_embeds, attention_mask=mask_cls)
        last_hidden_state = bert_out.last_hidden_state
        cls = last_hidden_state[:, 1]
        output = self.dropout(cls)
        output = self.out_linear_layer(output)
        return output

    def get_mask_output(self, tids, inputs_embeds, mask_prompt):
        bs = inputs_embeds.shape[0]
        bert_out = self.bert_model(inputs_embeds=inputs_embeds, attention_mask=mask_prompt)
        # bert_out = self.bert_model(input_ids=tids, attention_mask=mask_prompt, output_attentions=True) # visualization
        # layer * (bs, num_heads, seq_len, seq_len)
        sequence_output = bert_out.last_hidden_state
        mask_idx = (tids == 103).nonzero(as_tuple=True)[1]
        mask_output = sequence_output[t.arange(bs), mask_idx]
        mask_output = self.dropout(mask_output)
        prediction_mask_scores = self.prompt_linear_layer(mask_output)
        logits = []
        for label_tid in self.label_list:
            logits.append(prediction_mask_scores[:, label_tid].unsqueeze(-1))
        logits = t.cat(logits, -1)
        return logits
        # prediction_mask_scores = self.prompt_logits_linear_layer(mask_output)
        # return prediction_mask_scores

    def forward(self, tids, masks, vis1, vis2):
        mask0 = masks[:, 0]  # prompt
        mask1 = masks[:, 1]  # cls 1
        mask2 = masks[:, 2]  # cls 2
        bs = tids.shape[0]
        input_embeds = self.bert_model.embeddings(input_ids=tids)
        vis_prefix1 = self.vision_mapping(vis1).reshape(-1, self.prefix, self.bert_config.hidden_size)
        vis_prefix2 = self.vision_mapping(vis2).reshape(-1, self.prefix, self.bert_config.hidden_size)
        vis_prefixes = t.cat([vis_prefix1, vis_prefix2], dim=1).reshape(-1, self.bert_config.hidden_size)
        vis_idx_0 = (tids == self.vis_token).nonzero(as_tuple=True)[0]
        vis_idx_1 = (tids == self.vis_token).nonzero(as_tuple=True)[1]
        input_embeds[vis_idx_0, vis_idx_1] = vis_prefixes
        prompt_logits = self.get_mask_output(tids, input_embeds, mask0)
        cls1 = self._get_cls_output(input_embeds, mask1)
        cls2 = self._get_cls2_output(input_embeds, mask2)
        return prompt_logits, cls1, cls2
