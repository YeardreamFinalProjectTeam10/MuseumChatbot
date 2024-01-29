from transformers import AutoTokenizer, AutoModel, BertModel, BertTokenizerFast, Trainer, TrainingArguments
from kobert_tokenizer import KoBERTTokenizer
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()

        self.tokenizer = tokenizer
        model_name = tokenizer.name_or_path

        self.question_encoder = BertModel.from_pretrained(model_name)
        self.passage_encoder = BertModel.from_pretrained(model_name)

        self.passage_dropout = nn.Dropout(0.1)
        self.question_dropout = nn.Dropout(0.1)

        self.m = nn.LogSoftmax(dim=1)

    def forward(self, q_input_ids, q_attention_mask, q_token_type_ids,
                p_input_ids, p_attention_mask, p_token_type_ids):

        question_output = self.question_encoder(input_ids=q_input_ids, attention_mask=q_attention_mask, token_type_ids=q_token_type_ids)
        passage_output = self.passage_encoder(input_ids=p_input_ids, attention_mask=p_attention_mask, token_type_ids=p_token_type_ids)

        question_output = self.question_dropout(question_output.last_hidden_state)
        passage_output = self.passage_dropout(passage_output.last_hidden_state)

        question_embeddings = torch.stack([question_output[i][0] for i in range(len(question_output))])
        passage_embeddings = torch.stack([passage_output[i][0] for i in range(len(passage_output))])

        return question_embeddings, passage_embeddings