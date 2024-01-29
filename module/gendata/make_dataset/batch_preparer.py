import torch
from torch.utils.data import Dataset
from datasets import load_from_disk

class BatchPreparer(Dataset):
    def __init__(self, data_dir, tokenizer):
        self.tokenizer = tokenizer
        self.dataset = load_from_disk(data_dir)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        feature = self.prepare_features(data)
        return feature

    def prepare_features(self, data):
        q_key = 'pos_question' if 'pos_question' in data else 'question'
        n_key = 'neg_passage' if 'neg_passage' in data else 'hard_neg_passage'
        
        # Tokenization
        tokenized_question = self.tokenizer(data[q_key], truncation=True, padding='max_length', max_length=128, return_tensors="pt")
        tokenized_passage = self.tokenizer([data['pos_passage'], data[n_key]], truncation=True, padding='max_length', max_length=512, return_tensors="pt")
        
        # Merging features
        feature = {
            'input_ids': torch.cat((tokenized_question['input_ids'], tokenized_passage['input_ids']), dim=1),
            'attention_mask': torch.cat((tokenized_question['attention_mask'], tokenized_passage['attention_mask']), dim=1),
            'token_type_ids': torch.cat((tokenized_question['token_type_ids'], tokenized_passage['token_type_ids']), dim=1)
        }
        return feature
