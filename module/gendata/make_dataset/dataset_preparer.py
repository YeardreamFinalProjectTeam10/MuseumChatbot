import json
from tqdm import tqdm
import pandas as pd
from datasets import Dataset
from transformers import BertTokenizerFast

from module.gendata.make_dataset.custom_bm25 import CustomBM25

class DatasetPreparer:
    def __init__(self, file_name, tokenizer, neg_num):
        self.ctx_ids, self.contexts, self.questions = self.load_data(file_name)
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer)
        self.neg_num = neg_num
        self.bm25 = CustomBM25([self.tokenizer.encode(passage, add_special_tokens=False) for passage in self.contexts])

    def load_data(self, file_name):
        ctx_ids, contexts, questions = [], [], []
        with open(file_name, 'r') as file:
            for line in tqdm(file, desc='Loading data'):
                data = json.loads(line)
                if data['ctx_id'] not in ctx_ids:
                    ctx_ids.append(data['ctx_id'])
                    contexts.append(data['context'])
                    questions.append([])
                questions[-1].append(data['question'])
        return ctx_ids, contexts, questions

    def prepare_dataset(self):
        data = {'question': [], 'pos_passage': [], 'hard_neg_passage': []}
        for row, questions in enumerate(tqdm(self.questions, desc='Preparing dataset')):
            for col, _ in enumerate(questions):
                top_ctx_ids = self.get_negative_passage_bm25(row, col)
                real_ctx_idx = self.ctx_ids[row]
                if real_ctx_idx in top_ctx_ids:
                    top_ctx_ids.remove(real_ctx_idx)
                for idx in [self.ctx_ids.index(id) for id in top_ctx_ids]:
                    data['question'].append(self.questions[row][col])
                    data['pos_passage'].append(self.contexts[row])
                    data['hard_neg_passage'].append(self.contexts[idx])
        return Dataset.from_pandas(pd.DataFrame(data))

    def get_negative_passage_bm25(self, question_row, question_column):
        tokenized_question = self.tokenizer.encode(self.questions[question_row][question_column], add_special_tokens=False)
        top_ctx_idx = self.bm25.get_top_n(query=tokenized_question, ctxids=self.ctx_ids, neg_num=self.neg_num)
        return top_ctx_idx

def main():
    # 파일 이름과 토크나이저 설정
    file_name = "data/files/train.jsonl"  # 데이터셋 파일 경로
    tokenizer_name = "klue/bert-base"  # 사용할 토크나이저
    neg_num = 5  # negative passage 생성 갯수
    batch_size = 4  # 배치 크기

    # BertTokenizerFast 인스턴스 초기화
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)

    # DatasetPreparer 인스턴스 생성 및 데이터셋 준비
    dataset_preparer = DatasetPreparer(file_name, tokenizer, neg_num)
    prepared_dataset = dataset_preparer.prepare_dataset()

    # Optional: Prepared_dataset을 디스크에 저장
    data_dir = "data/files/dataset"
    prepared_dataset.save_to_disk(data_dir)

if __name__ == "__main__":
    main()
