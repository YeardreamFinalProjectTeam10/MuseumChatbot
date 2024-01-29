from module.gendata.question.generate_question import GenerateQuestion
from module.gendata.question.question_preprocessing import PreprocessingQuestion
from module.gendata.save_data import save_jsonl, save_csv

def main():
    # Step 1: Generate Questions
    jsonl_input_path = "data/preprocessed/bert_split.jsonl"  # Update this to your actual input JSONL file path
    question_generator = GenerateQuestion(jsonl_input_path=jsonl_input_path)
    generated_questions_df = question_generator.generate_question()

    # Optional: Save the raw generated questions to a CSV file
    raw_questions_output_path = "data/question/raw_questions" # Omit the .csv
    save_csv(generated_questions_df, raw_questions_output_path)

    # Step 2: Preprocess Questions
    preprocessor = PreprocessingQuestion(dataset=generated_questions_df)
    preprocessed_df = preprocessor.format_question()

    # Step 3: Split into Train and Test sets
    train_df, test_df = preprocessor.train_test_split(df=preprocessed_df, test_size=0.2, random_state=42)

    # Step 4: Save Train and Test Data
    train_output_path = "data/files/train" # Omit the .jsonl
    test_output_path = "data/files/test" # Omit the .jsonl
    save_jsonl(train_df, train_output_path)
    save_jsonl(test_df, test_output_path)

    print("Data processing complete. Train and test datasets are saved.")

if __name__ == "__main__":
    main()