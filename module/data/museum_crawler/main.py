from driver_utils import DriverUtils
from scrapy import Scrapy
from save_data  import save_data, save_log_data

import pandas as pd


def main():
    driver = DriverUtils.initialize_driver()
    scraper = Scrapy(driver)
    last_page = scraper.get_last_page()
    print(f'크롤링 대상 전체 페이지: {last_page}p')

    all_data = []
    for page_number in range(1, last_page + 1):
        page_data = scraper.scrape_page_data(page_number)
        all_data.extend(page_data)

    # 파일 위치 : module/data/files/museum_passage.csv/jsonl
    save_data(all_data, 'museum_passage')

if __name__ == "__main__":
    main()
