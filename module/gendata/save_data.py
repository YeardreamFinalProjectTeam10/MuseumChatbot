import pandas as pd

def save_jsonl(data, filepath):
    df = pd.DataFrame(data)
    df.to_json(f"{filepath}.jsonl", orient='records', lines=True, force_ascii=False)

def save_csv(data, filepath):
    df = pd.DataFrame(data)
    df.to_csv(f'{filepath}.csv', encoding='utf-8-sig')
