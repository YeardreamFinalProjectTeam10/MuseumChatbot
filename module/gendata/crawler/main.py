from module.gendata.crawler.driver_utils import DriverUtils
from module.gendata.crawler.scrapy import Scrapy
from module.gendata.save_data import save_jsonl

def main():
    driver = DriverUtils.initialize_driver()
    scraper = Scrapy(driver)
    last_page = scraper.get_last_page()
    print(f'크롤링 대상 전체 페이지: {last_page}p')

    all_data = []
    for page_number in range(1, last_page + 1):
        page_data = scraper.scrape_page_data(page_number)
        all_data.extend(page_data)

    save_jsonl(all_data, 'data/crawled/museum_passage')

if __name__ == "__main__":
    main()
