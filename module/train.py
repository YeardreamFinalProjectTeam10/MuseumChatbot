import argparse
import json
import os
from attrdict import AttrDict

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertTokenizerFast, get_linear_schedule_with_warmup
from kobert_tokenizer import KoBERTTokenizer

from datasets import load_from_disk
from tqdm import tqdm
import numpy as np

from module.gendata.make_dataset.batch_preparer import BatchPreparer
from model import Model
from module.inference import Inference

class DPRTrainer:
    def __init__(self, args):
        self.args = args
        self.setup_seed(args.random_seed)
        self.device = self.setup_device(args.gpu_ids)
        self.tokenizer = self.init_tokenizer(args.tokenizer)
        self.model = self.init_model().to(self.device)
        self.train_dataset, self.eval_dataset = self.init_datasets()
        self.train_dataloader, self.eval_dataloader = self.init_dataloaders()
        self.optimizer, self.scheduler, self.loss_fn = self.setup_training()

    def setup_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_device(self, gpu_ids):
        return torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")

    def init_tokenizer(self, tokenizer_name):
        if 'skt' in tokenizer_name:
            return KoBERTTokenizer.from_pretrained(tokenizer_name)
        else:
            return AutoTokenizer.from_pretrained(tokenizer_name)

    def init_model(self):
        model = Model(tokenizer=self.tokenizer)
        if -1 not in self.args.gpu_ids and len(self.args.gpu_ids) > 1:
            model = nn.DataParallel(model)
        return model

    def init_datasets(self):
        train_dataset = BatchPreparer(self.args.train_data_dir, self.tokenizer)
        eval_dataset = BatchPreparer(self.args.validation_data_dir, self.tokenizer)
        return train_dataset, eval_dataset

    def init_dataloaders(self):
        train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
        eval_dataloader = DataLoader(
            self.eval_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
        return train_dataloader, eval_dataloader

    def setup_training(self):
        self.iterations = len(self.train_dataset) // self.args.batch_size
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=self.iterations * self.args.num_epochs)
        loss_fn = nn.NLLLoss()
        return optimizer, scheduler, loss_fn

    def train_one_epoch(self):
        self.model.train()

        for i, batch in tqdm(enumerate(self.train_dataloader)):
            question, passage=batch

            q_input_ids = question['input_ids']
            q_input_ids = q_input_ids.to(self.device)

            q_attention_mask = question['attention_mask']
            q_attention_mask = q_attention_mask.to(self.device)

            q_token_type_ids = question['token_type_ids']
            q_token_type_ids = q_token_type_ids.to(self.device)

            p_input_ids = passage['input_ids']
            p_input_ids = p_input_ids.to(self.device)

            p_attention_mask = passage['attention_mask']
            p_attention_mask = p_attention_mask.to(self.device)

            p_token_type_ids = passage['token_type_ids']
            p_token_type_ids = p_token_type_ids.to(self.device)

            question_output, passage_output = self.model(q_input_ids, q_attention_mask, q_token_type_ids, p_input_ids, p_attention_mask, p_token_type_ids)

            output = self.m(torch.matmul(question_output, passage_output.T))

            #target = torch.tensor([i for i in range(question['input_ids'].size(0))])
            target = torch.tensor([2*i for i in range(self.args.batch_size)])
            target = target.to(self.device)

            loss = self.loss_fn(output, target)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

    def evaluate(self):
        self.model.eval()
        total_loss = 0.
        correct = 0
        correct_3 = 0
        cnt = 0
        with torch.no_grad():
            for batch in self.eval_dataloader:
                question, passage = batch

                q_input_ids = question['input_ids']
                q_input_ids = q_input_ids.to(self.device)

                q_attention_mask = question['attention_mask']
                q_attention_mask = q_attention_mask.to(self.device)

                q_token_type_ids = question['token_type_ids']
                q_token_type_ids = q_token_type_ids.to(self.device)

                p_input_ids = passage['input_ids']
                p_input_ids = p_input_ids.to(self.device)

                p_attention_mask = passage['attention_mask']
                p_attention_mask = p_attention_mask.to(self.device)

                p_token_type_ids = passage['token_type_ids']
                p_token_type_ids = p_token_type_ids.to(self.device)

                question_output, passage_output = self.model(q_input_ids, q_attention_mask, q_token_type_ids,
                                                             p_input_ids, p_attention_mask, p_token_type_ids)

                output = self.m(torch.matmul(question_output, passage_output.T))

                # target = torch.tensor([i for i in range(question['input_ids'].size(0))])
                target = torch.tensor([2 * i for i in range(self.args.batch_size)])
                target = target.to(self.device)

                loss = self.loss_fn(output, target)
                total_loss += loss

                #정확도 계산
                for i in range(question['input_ids'].size(0)):
                    values, indices = torch.topk(output[i], k=10)
                    cnt+=1
                    if (2*i) in indices:
                        correct+=1

                    values, indices = torch.topk(output[i], k=3)
                    if (2 * i) in indices:
                        correct_3 += 1

        return total_loss / len(self.eval_dataloader), correct/cnt, correct_3/cnt

    def train(self):
        try:
            self.model.load_state_dict(torch.load(self.args.save_dirpath + self.args.load_pthpath + 'model.pth'))
            print("Loaded pre-trained model successfully.")
        except FileNotFoundError:
            print("Pre-trained model not found. Training from scratch.")
    
        for e in tqdm(range(self.args.num_epochs)):
            print('Epoch:', e)
            self.train_one_epoch()
            val_loss, val10_acc, val3_acc = self.evaluate()
            print('val_total_loss', val_loss)
            print('val_acc_about_top10', val10_acc)
            print('val_acc_about_top3', val3_acc)
            self.scores.append(val_loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model = self.model
                torch.save(self.best_model.state_dict(), self.args.save_dirpath + self.args.load_pthpath + 'model.pth')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="DPR-National Museum of Korea")
    arg_parser.add_argument("--mode", dest="mode", type=str, default="train")
    arg_parser.add_argument("--config_dir", dest="config_dir", type=str, default="config", help="Config Directory")
    arg_parser.add_argument("--config_file", dest="config_file", type=str, default="museum",
                            help="Config json file")
    parsed_args = arg_parser.parse_args()

    # Read from config file and make args
    with open(os.path.join(parsed_args.config_dir, "{}.json".format(parsed_args.config_file))) as f:
        args = AttrDict(json.load(f))

    # print("Training/evaluation parameters {}".format(args))
    dpr_train = DPRTrainer(args)
    if parsed_args.mode == 'train':
        dpr_train.train()
    else:
        inf = Inference(args, dpr_train.model)
        top_5, top_20, top_100 = inf.test_retriever()
        print("top_5", top_5)
        print("top_20", top_20)
        print("top_100", top_100)