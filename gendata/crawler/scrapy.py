from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.webdriver.common.by import By

from tqdm import tqdm
import time
import random

class Scrapy:
    '''
    Desc:
        실제 크롤링을 실행하는 클래스
    Args:
        DriverUtils에서 return한 driver object
    '''
    def __init__(self, driver):
        self.driver = driver
    
    # 큐레이터 추천소장품 카테고리의 마지막 페이지 return
    def get_last_page(self):
        base_url = 'https://www.museum.go.kr/site/main/relic/recommend/list'
        self.driver.get(base_url)
        time.sleep(2)
        last_page = self.driver.find_element(By.CLASS_NAME, 'allPage').text.split(' ')[3]
        return int(last_page)

    # 페이지별 데이터 수집
    def scrape_page_data(self, page_number):
        page_data = []
        page_url = f'https://www.museum.go.kr/site/main/relic/recommend/list?cp={page_number}'

        self.driver.execute_script("window.open('');")
        self.driver.switch_to.window(self.driver.window_handles[-1])
        time.sleep(random.randint(1, 3))
        # 해당 페이지로 이동
        self.driver.get(page_url)
        time.sleep(random.randint(2, 6))
        try:
            soup = bs(self.driver.page_source, "html.parser")
            cards = soup.find_all("li", class_="card")
            for card in tqdm(cards, desc=f"{page_number}페이지 진행상황"):
                relic_data = self.parse_relic_data(card)
                if relic_data:
                    page_data.append(relic_data)
        except Exception as e:
            print(f'페이지 내 정보 받아오기 실패 -> {e}')
            print(f'오류 페이지 : {page_url}')
            raise ValueError('Failed to retrieve list items from the page')
        self.driver.close()
        self.driver.switch_to.window(self.driver.window_handles[-1])
        return page_data

    def parse_relic_data(self, card):
        parsed_data = []
        try:
            tag_a = card.find("a", class_="img-box")
            title = card.find('div', class_='k-cura-txt').find('a').get_text()
            country_and_era = card.find('strong', string='국적/시대').find_next_sibling('p').get_text(strip=True)
            size_and_category = card.find('strong', string='크기/지정구분').find_next_sibling('p').get_text(separator='\n', strip=True)
            href = tag_a['href']
            link = f"https://www.museum.go.kr{href}"

            self.driver.execute_script("window.open('');")
            time.sleep(1)
            self.driver.switch_to.window(self.driver.window_handles[-1])
            time.sleep(random.randint(2, 5))

            # 상세페이지 이동
            self.driver.get(link)
            time.sleep(3)
            soup = bs(self.driver.page_source, 'html.parser')
            # h5 : 소제목, p : 본문
            titles_and_paragraphs = soup.select('.prg > h5, .prg > p')
            # 소제목 초기화
            current_subtitle = ''
            # 소제목이 나오면 아예 새로운 딕셔너리 데이터 추가 후 개행문자 추가
            for item in titles_and_paragraphs:
                if item.name == 'h5':
                    current_subtitle = item.get_text(strip=True)
                    if current_subtitle:
                        parsed_data.append({
                            'title': title,
                            'era': country_and_era,
                            'info': size_and_category,
                            'description': '소제목:' + current_subtitle + '\n'
                        })
                # 본문이 나오고 저장된 소제목이 있으면 새로 행 추가 없이 마지막 description에 내용 추가
                elif item.name == 'p' and current_subtitle:
                    parsed_data[-1]['description'] += item.get_text(strip=True)
                # 본문이 나왔으나 지정된 소제목이 없는 경우
                elif item.name == 'p' and not current_subtitle:
                    parsed_data.append({
                            'title' : title,
                            'era' : country_and_era,
                            'info' : size_and_category,
                            'description' : item.get_text(strip=True)
                        })

            self.driver.close()
            self.driver.switch_to.window(self.driver.window_handles[-1])
            time.sleep(2)

            return parsed_data

        except Exception as e:
            print(f'상세페이지 내 오류 발생 --> {e}')
            save_log_data(parsed_data, 'error_log')
            return parsed_data