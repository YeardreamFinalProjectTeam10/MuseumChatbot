from dotenv import load_dotenv
import os
from tqdm import tqdm
import pandas as pd
import jsonlines

import openai

class GenerateQuestion :
    def __init__(self,
                 jsonl_input_path: str,
                 title: str = "title",
                 description: str = "description") :
        '''
            Args:
                jsonl_input_path : 질문을 생성할 기반 문서 (JSONL) 파일의 경로, 500 토큰 미만으로 Split 할 것을 권장.
                title : 데이터 내 "제목"의 필드값 (key)
                description : 데이터 내 "큐레이션 설명"의 필드값 (key)
        '''
        self.jsonl_input_path = jsonl_input_path
        self.title = title
        self.description = description

        # TODO: .env에 {OPENAI_API_KEY=...} 형식의 openai key가 필요함
        load_dotenv()
        openai.api_key = os.getenv('OPENAI_API_KEY')

        # API KEY가 설정되지 않았으면 에러 발생시킴
        if not openai.api_key:
            raise ValueError(".env에 OPENAI_API_KEY가 없습니다.")
    
    def generate_question(
            self,
            model='gpt-3.5-turbo-1106',
            max_retries=3
        ):

        dataset = []
        with jsonlines.open(self.jsonl_input_path) as file:
            for data in tqdm(file.iter(), desc="Generating Questions"):
                try :
                    title = data[self.title]
                    description = data[self.description]
                    
                    # 요청 보내기
                    response = openai.ChatCompletion.create(
                        model = model,
                        messages=[
                            {'role': 'system', 'content': '이 대화에서는 주어진 유물 해설을 바탕으로 관람객이 던질 만한 질문을 생성합니다. 질문은 문서에 직접적으로 언급된 키워드를 사용하지 않고, 간접적으로 묻는 방식이어야 합니다. 또한, 반드시 문서 내용을 기반으로 답변할 수 있는 질문이어야 합니다.'},
                            {'role': 'user', 'content': f'문서의 해설: {description}'},
                            {'role': 'system', 'content': '이제 문서에 근거하여 관람객이 물을 만한 질문과 그에 대한 답변을 생성하십시오. 답변 형식은 "question: 질문\nanswer: 답변"이어야 합니다.'}
                        ]
                    )

                    # 재시도 3회까지 허용
                    retries = 0
                    while retries <= max_retries :
                        print(f'{retries}회 시도 중..')
                        if response.choices and response.choices[0].message['content'] :
                            try :
                                qa_pair = response.choices[0].message['content'].strip()
                                retries = max_retries+1
                            except Exception as e:
                                print(f"{e}\n 메시지 생성 후 strip 실패, 재시도 중 ...")
                                retries += 1
                                if retries > max_retries:
                                    print(f"{title}: strip 3회 이상 실패... 다음 질문으로 넘어갑니다.")
                                    qa_pair = '실패'
                                    continue
                        else :
                            retries += 1
                            if retries > max_retries:
                                print(f"{title}: 생성 3회 이상 실패... 다음 질문으로 넘어갑니다.")
                                qa_pair = '실패'
                                continue
                            print(f"생성 실패.. {retries}/{max_retries}회 에러")

                except Exception as timeout:
                    qa_pair='실패'
                    print(f'TimeOut, 전체 실패. 다음 Passage를 처리합니다. --> {timeout}')
                    continue
                dataset.append({'title': title, 'context': description, 'question': qa_pair})
        
        df = pd.DataFrame(dataset)

        return df