import pandas as pd
import numpy as np
import json
import csv
import glob

class PreprocessingQuestion:
    '''
    A class for preprocessing GPT API-generated question datasets.
    '''
    def __init__(self, dataset=None, input_csv_path: str = None):
        if dataset is not None:
            self.dataset = dataset
        elif input_csv_path:
            try:
                self.dataset = pd.read_csv(input_csv_path)
            except Exception as e:
                print(f"Error loading CSV file: {e}")
                self.dataset = pd.DataFrame()  # Initialize an empty DataFrame if loading fails
        else:
            print("No dataset or CSV path provided, initializing with an empty DataFrame.")
            self.dataset = pd.DataFrame()


    def format_question(self):
        '''
        Desc:
            1. Assign ctx_id and tit_id,
                - ctx_id : 동일한 passage에 붙는 고유 id
                - tit_id : 동일한 title에 붙는 고유 id
                - 참고 : 하나의 title(주제)에는 여러 개로 split된 passage(문단)들이 있음
            2. Separate the columns for questions and answers.
        '''
        df = self.dataset.copy()
        
        df['ctx_id'] = df.index
        df['tit_id'] = df.groupby('title').ngroup()
        df['question'], df['answer'] = df['question'].str.split('질문:', 1).str[1].str.split('답변:', 1).str

        # Cleaning up questions and answers
        for col in ['question', 'answer', 'context']:
            df[col] = df[col].str.strip(',. ')
        df.dropna(subset=['answer'], inplace=True)

        return df

    def train_test_split(
            self, df=None, input_path=None,
            test_size=0.2, random_state=10):
        
        '''
        Save train and test csv files.
        '''
        if df is None:
            print("DataFrame is not provided. Attempting to load from input_path.")
            try :
                df = pd.read_csv(input_path)
            except Exception as e:
                print(f'Error loading input CSV Files: {e}')
                return

        # Ensure unique contexts are split appropriately
        ctx_counts = df['ctx_id'].value_counts()
        eligible_for_test = ctx_counts[ctx_counts >= 2].index

        # Randomly select contexts for the test set
        np.random.seed(random_state)
        test_ctx_ids = np.random.choice(eligible_for_test, size=int(df.shape[0] * test_size), replace=False)

        is_test = df['ctx_id'].isin(test_ctx_ids)
        train_df = df[~is_test]
        test_df = df[is_test]

        return train_df, test_df