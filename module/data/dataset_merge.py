import random
import argparse

import torch
from torch.utils.data import Dataset
from datasets import load_from_disk
from datasets import Dataset as data_set
from datasets import load_dataset, concatenate_datasets

import pyarrow as pa
import pyarrow.parquet as pq

class MergeDataset:
    def __init__(self, dataset_array):
        self.dataset_array = dataset_array
        self.questions = []
        self.pos_passages = []
        self.hard_neg_passages = []

    def only_pos_make_dataset(self):
        for dataset in self.dataset_array:
                for data in dataset:
                    if 'pos_question' in data and data['pos_question'] :
                        question = data['pos_question']
                        if question not in self.questions :                            
                            self.questions.append(data['pos_question'])
                            self.pos_passages.append(data['pos_passage'])
                    else :
                        question = data['question']
                        if question not in self.questions :                            
                            self.questions.append(data['question'])
                            self.pos_passages.append(data['pos_passage'])

        return data_set.from_dict({'question': self.questions,
                                  'pos_passage': self.pos_passages})

    def pos_neg_make_dataset(self):
        for dataset in self.dataset_array:
                for data in dataset:
                    if 'pos_question' in data and data['pos_question'] :
                        self.questions.append(data['pos_question'])
                        self.pos_passages.append(data['pos_passage'])
                        self.hard_neg_passages.append(data['neg_passage'])
                    else :
                        self.questions.append(data['question'])
                        self.pos_passages.append(data['pos_passage'])
                        self.hard_neg_passages.append(data['hard_neg_passage'])

        return data_set.from_dict({'question': self.questions,
                                  'pos_passage': self.pos_passages,
                                  'hard_neg_passage': self.hard_neg_passages})


def main(mode, dataset1: str = "data/files/museum_train", dataset2: str = "data/files/kdpr_train"):
    data1 = load_from_disk(dataset1)
    data2 = load_from_disk(dataset2)
    merged_data = MergeDataset([data1, data2])

    if mode == 'p':
        # Only-Positive-Passage 데이터셋 생성 및 저장
        output_only_pos_dataset_path = "data/files/merged_only_pos_dataset"
        pos_merge_train = merged_data.only_pos_make_dataset()
        pos_merge_train.save_to_disk(output_only_pos_dataset_path)
    
    elif mode == 'pn':
        # Positive & Negative Passage 데이터셋 생성 및 저장
        output_pos_neg_dataset_path = "data/files/merged_pos_neg_dataset"
        pos_neg_merge_train = merged_data.pos_neg_make_dataset()
        pos_neg_merge_train.save_to_disk(output_pos_neg_dataset_path)

    else:
        raise ValueError("Invalid mode. Choose 'p' for only positive passages or 'pn' for positive and negative passages.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge datasets for DPR train")
    parser.add_argument('--mode', type=str, required=True, 
                        help="Positive Passage만 Merge하려면 'p' or Negetive와 Positive 모두 Merge하려면 'pn'을 입력하세요.")
    args = parser.parse_args()
    main(args.mode)