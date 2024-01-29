import torch
from torch import nn
import faiss
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertTokenizerFast
from kobert_tokenizer import KoBERTTokenizer

from module.gendata.make_dataset.batch_preparer import BatchPreparer

class Inference(object):
    def __init__(self, args, model, save_dir=None):
        self.args = args
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda:" + str(self.args.gpu_ids[0]) if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.load_state_dict(torch.load(self.args.save_dirpath + self.args.load_pthpath + 'model.pth'))   
        self.model = self.model.to(self.device)
        self.tokenizer = self.init_tokenizer(args.tokenizer)
        self.passage_dropout =nn.Dropout(0.1)
        self.question_dropout = nn.Dropout(0.1)

        print(
            """
            # -------------------------------------------------------------------------
            #   LOAD DPR MODEL DONE
            # -------------------------------------------------------------------------
            """
        )

        self.eval_dataset = BatchPreparer(self.args.validation_data_dir, self.tokenizer)

        self.dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.cpu_workers,
            drop_last=True,
            collate_fn=self.eval_dataset.collate_fn
        )

        if save_dir is None:

            print(
                """
                # -------------------------------------------------------------------------
                #   MAKE INDEX START
                # -------------------------------------------------------------------------
                """
            )
            self.index = faiss.IndexFlatIP(768)
            self.index = faiss.IndexIDMap2(self.index)

            with torch.no_grad() :
                cnt = 0
                for i, batch in tqdm(enumerate(self.dataloader)):
                    question, passage = batch

                    p_input_ids = passage['input_ids']
                    p_input_ids = p_input_ids.to(self.device)

                    p_attention_mask = passage['attention_mask']
                    p_attention_mask = p_attention_mask.to(self.device)

                    p_token_type_ids = passage['token_type_ids']
                    p_token_type_ids = p_token_type_ids.to(self.device)

                    encoded_passage = self.model.module.passage_encoder(input_ids=p_input_ids, attention_mask=p_attention_mask, token_type_ids=p_token_type_ids)

                    passage_output = self.passage_dropout(encoded_passage.last_hidden_state)
                    passage_embeddings = torch.stack([passage_output[i][0] for i in range(len(passage_output))]).cpu().detach().numpy()

                    self.index.add_with_ids(passage_embeddings.astype('float32'), np.array(range(cnt, cnt + passage_embeddings.shape[0])))

                    cnt += passage_embeddings.shape[0]

            faiss.write_index(self.index, 'museum_index')

        else:
            self.index = faiss.read_index(save_dir)

    def init_tokenizer(self, tokenizer_name):
        if 'skt' in tokenizer_name:
            return KoBERTTokenizer.from_pretrained(tokenizer_name)
        else:
            return AutoTokenizer.from_pretrained(tokenizer_name)

    def test_retriever(self):
        self.model.eval()

        correct_2 = 0
        correct_5 = 0
        correct_10 = 0
        correct_20 = 0
        correct_100 = 0

        cnt_2 = 0
        cnt_5 = 0
        cnt_10 = 0
        cnt_20 = 0
        cnt_100 = 0
        
        with torch.no_grad():
            for batch in self.dataloader:
                question, passage = batch

                q_input_ids = question['input_ids']
                q_input_ids = q_input_ids.to(self.device)

                q_attention_mask = question['attention_mask']
                q_attention_mask = q_attention_mask.to(self.device)

                q_token_type_ids = question['token_type_ids']
                q_token_type_ids = q_token_type_ids.to(self.device)

                encoded_question = self.model.module.question_encoder(input_ids=q_input_ids, attention_mask=q_attention_mask, token_type_ids=q_token_type_ids)

                question_output = self.question_dropout(encoded_question.last_hidden_state)
                question_embeddings = torch.stack([question_output[i][0] for i in range(len(question_output))]).cpu().detach().numpy().astype('float32')

                distances, indices = self.index.search(question_embeddings, 100)

                # top 100
                for idxs in indices:
                    if (cnt_100*2) in idxs:
                        correct_100 +=1
                        cnt_100 +=1
                    else:
                        cnt_100 += 1

                #top 20
                for idxs in indices:
                    if (cnt_20*2) in idxs[:20]:
                        correct_20 +=1
                        cnt_20 +=1
                    else:
                        cnt_20 += 1

                #top 10
                for idxs in indices:
                    if (cnt_10*2) in idxs[:10]:
                        correct_10 +=1
                        cnt_10 +=1
                    else:
                        cnt_10 += 1

                #top 5
                for idxs in indices:
                    if (cnt_5*2) in idxs[:5]:
                        correct_5 +=1
                        cnt_5 +=1
                    else:
                        cnt_5 += 1

                #top 2
                for idxs in indices:
                    if (cnt_2*2) in idxs[:2]:
                        correct_2 +=1
                        cnt_2 +=1
                    else:
                        cnt_2 += 1

        return correct_2/cnt_2, correct_5/cnt_5, correct_10/cnt_10, correct_20/cnt_20, correct_100/cnt_100

    

        



        


