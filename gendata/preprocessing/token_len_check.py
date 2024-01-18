
def bert_len_check(encoding_name: str = "klue/bert-base", text: str):
    """
    Desc:
        텍스트를 입력하면 BERT 인코딩 기준으로 토큰 갯수를 반환합니다.
    Args:
        text (str): 검사할 텍스트
        encoding_name: BERT 모델 이름
    """
    tokenizer = BertTokenizerFast.from_pretrained(encoding_name)
    tokens = tokenizer(text, return_tensors='pt')['input_ids']
    print(f'해당 텍스트의 토큰 길이 (BERT) : {tokens.size(1)}')

    return tokens.size(1)

    
def tiktoken_len_check(encoding_name: str = "cl100k_base", text: str):
    """
    Desc:
        텍스트를 입력하면 GPT 인코딩 기준으로 토큰 갯수를 반환합니다.
    Args:
        text (str): 검사할 텍스트
        encoding_name: 토큰화에 사용할 모델 이름
            - gpt-3.5-turbo, gpt-4, text-embedding-ada-002 : "cl100k_base" (default)
            - Codex models, text-davinci-002, text-davinci-003 : "p50k_base"
            - GPT-3 models like "davinci" : "r50k_base" or "gpt2"
    """
    tokenizer = tiktoken.get_encoding(self.encoding_name)
    tokens = tokenizer.encode(text)
    print(f'해당 텍스트의 토큰 길이 (GPT) : {len(tokens)}')
    return len(tokens)