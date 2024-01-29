from module.gendata.save_data import save_jsonl
from module.gendata.preprocessing.crawled_data_preprocessing import DataExtractor, DescCombiner
from module.gendata.preprocessing.text_splitter import TextSplitter

def main():
    # Crawled data extraction
    data_extractor = DataExtractor(jsonl_input_path='data/crawled/museum_passage.jsonl')
    df = data_extractor.extract_data()

    # Combine data
    desc_combiner = DescCombiner()
    df = desc_combiner.combine_data(df)

    # Text Splitting
    # jsonl_input_path은 Optional, 넣으려면 꼭 desc_combine처리 완료된 jsonl 파일만 넣을 것.
    text_splitter = TextSplitter(
        jsonl_input_path=None,
        encoding_name='klue/bert-base',
        split_type='bert',
        chunk_size=400,
        chunk_overlap=80
    )
    split_data = text_splitter.split(df)
    save_jsonl(split_data, 'data/preprocessed/bert_split')

if __name__ == "__main__":
    main()