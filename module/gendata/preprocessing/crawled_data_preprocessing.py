import pandas as pd
import json

class DataExtractor:
    def __init__(self, jsonl_input_path="data/crawled/museum_passage.jsonl"):
        self.jsonl_input_path = jsonl_input_path
    
    def extract_data(self):
        data_list = []
        with open(jsonl_input_path, 'r', encoding='utf-8') as file:
            for line in file:
                json_data = json.loads(line)
                data_list.append({
                    'title': json_data['title'],
                    'era': json_data['era'],
                    'info': json_data['info'],
                    'description': json_data['description']
                })
        return pd.DataFrame(data_list)

class DescCombiner:
    @staticmethod
    def combine_data(self, df: pd.DataFrame):
        # info column split (by \n)
        info_df = df['info'].str.split('\\n', expand=True)
        info_df.columns = ['size', 'property', 'num']

        # replace if 'num' is empty and 'property' is not empty
        mask = (info_df['num'].isna()) & (info_df['property'].notna())
        info_df.loc[mask, 'num'] = info_df['property']
        info_df.loc[mask, 'property'] = None

        # Merge with the source dataframe
        df = pd.concat([df, info_df], axis=1)
        df.fillna('정보없음', inplace=True)

        # ['era', 'size', 'property', 'number'] only needs to be present once per title
        info_df = df.groupby('title').first().reset_index()
        info_df['description'] = info_df.apply(lambda row: f"작품명:{row.title}, 국적/시대:{row.era}, 크기:{row.size}, 문화재구분:{row.property}, 소장품번호:{row.num}", axis=1)
        info_df.drop(columns='new_desc', axis=1, inplace=True)

        # Append info_df to the bottom row of the source df
        df = pd.concat([df, info_df])
        df = df.reset_index(drop=True)

        # Delete everything except title, desc
        df.drop(columns=['era', 'info', 'size', 'num', 'property'], axis=1, inplace=True)

        return df