import json
from transformers import BertTokenizerFast
import tiktoken
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

"""
데이터의 Description 부분을 BERT/GPT 토크나이저로 작게 분할할 때 사용
split : 입력된 길이로 split함 (파라미터로 입력)
len_check : 길이 검사할 때 사용 (return 없음)
"""
class TextSplitter:
    ''' 
    Args
        jsonl_input_path : DF를 받아서 사용하면 넣을 필요 없음
        encoding_name: 토큰화에 사용할 모델 이름
            BERT
            - 'klue/bert-base' (default)
            - 기타 bert 계열의 pretrained 모델 (BertTokenizerFast에서 사용 가능해야함) 사용 가능
            GPT
            - gpt-3.5-turbo, gpt-4, text-embedding-ada-002 : "cl100k_base" (default)
            - Codex models, text-davinci-002, text-davinci-003 : "p50k_base"
            - GPT-3 models like "davinci" : "r50k_base" or "gpt2"
    '''
    def __init__(self, jsonl_input_path=None, encoding_name='klue/bert-base', split_type='bert', chunk_size=400, chunk_overlap=80):
        self.jsonl_input_path = jsonl_input_path
        self.encoding_name = encoding_name
        self.split_type = split_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_jsonl_data(self):
        try:
            data_list = []
            with open(self.jsonl_input_path, 'r', encoding='utf-8') as file:
                for line in file:
                    data_list.append(json.loads(line))
            return data_list
        except FileNotFoundError:
            raise FileNotFoundError(f"{self.jsonl_input_path} 파일을 찾을 수 없습니다.")
        except IOError as e:
            raise IOError(f"{self.jsonl_input_path} 파일 읽기 중 오류가 발생했습니다: {e}")

    def split(self, df=None):
        # Bert, Gpt 둘 중에 하나만 사용 가능
        if self.split_type not in ['bert', 'gpt']:
            raise ValueError("split_type must be 'bert' or 'gpt'.")
        
        if self.split_type == 'bert':
            tokenizer = BertTokenizerFast.from_pretrained(self.encoding_name)
            splitter = CharacterTextSplitter.from_huggingface_tokenizer(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, tokenizer=tokenizer, separators=["한편", "이 밖에도", ". "], keep_separator=True)
        else:
            tokenizer = tiktoken.get_encoding(self.encoding_name)
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=2000, chunk_overlap=100)

        if df is None and self.jsonl_input_path is None:
            raise ValueError("Either 'df' or 'jsonl_input_path' must be provided.")

        # DF 입력 없을 시 파일 호출, 있으면 해당 DF를 jsonl 형태로 변경
        if df is None:
            data = load_jsonl_data(self.jsonl_input_path)
        else:
            data = df.to_dict(orient='records')
        
        split_data_list = []

        # description 키가 있는지 확인 먼저 하기
        for text_dict in data:
            try:
                text = text_dict['description']
            except KeyError:
                raise KeyError("DataFrame에서 'description' 열을 찾을 수 없습니다.")

            # BERT 기준 Split
            if self.split_type = 'bert':
                split_data = splitter.split_text(text)
                for single_split_data in split_data:
                    new_line = text_dict.copy()
                    new_line['description'] = single_split_data
                    split_data_list.append(new_line)
            
            # GPT 기준 Split
            else:
                # 2천 토큰 넘어가면 잘라주기 (GPT에도 제한 토큰 있음)
                token_len = len(tokenizer.encode(text))            
                if token_len > 2000:
                    split_data = splitter.split_text(text)
                    for single_split_data in split_data:
                        new_line = text_dict.copy()
                        new_line['description'] = single_split_data
                        split_data_list.append(new_line)
                else:
                    split_data_list.append(text_dict)
        return split_data_list

def len_check(self, text=None, df=None):
    if self.split_type not in ['bert', 'gpt']:
        raise ValueError("split_type must be 'bert' or 'gpt'.")

    if text is None and df is None and self.jsonl_input_path is None:
        raise ValueError("Either 'text', 'df', or 'jsonl_input_path' must be provided.")

    if self.split_type == 'bert':
        tokenizer = BertTokenizerFast.from_pretrained(self.encoding_name)
        if text is None:
            if df is None and self.jsonl_input_path is not None:
                data = load_jsonl_data(self.jsonl_input_path)
            else: # df가 있는 경우
                data = df.to_dict(orient='records')
            
            for text_dict in data:
                tokens = tokenizer(text_dict['description'], return_tensors='pt')['input_ids']
                print(f'"{text_dict["title"]}" 토큰 길이: {tokens.size(1)}')

        # text가 있는 경우
        else:
            tokens = tokenizer(text, return_tensors='pt')['input_ids']
            print(f'해당 텍스트의 토큰 길이: {tokens.size(1)}')
    
    # self.split_type == 'gpt'
    else:
        tokenizer = tiktoken.get_encoding(self.encoding_name)
        if text is None:
            if df is None and self.jsonl_input_path is not None:
                data = load_jsonl_data(self.jsonl_input_path)
            else:
                data = df.to_dict(orient='records')

            for text_dict in data:
                tokens = tokenizer.encode(text_dict['description'])
                print(f'"{text_dict["title"]}" 토큰 길이: {len(tokens)}')
        else :
            tokens = tokenizer.encode(text)
            print(f'해당 텍스트의 토큰 길이: {len(tokens)}')