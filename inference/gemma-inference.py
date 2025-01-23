from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import Dataset
import pandas as pd
import torch


def create_prompt(input_text):
    """
    망가진 글자를 복원시키기 위한 프롬프트.
    """
    return (
        "<start_of_turn> Your task is to transform the given obfuscated Korean review into a clear, correct, "
        "and natural-sounding Korean review that reflects its original meaning.\n"
        f"Input: {input_text}\n"
        "<end_of_turn>\n"
        "<start_of_turn>Assistant:\n"
        "Output:"
    )


def process_batch(batch, pipeline):
    """
    배치 데이터를 처리하는 함수.
    """
    prompts = [create_prompt(text) for text in batch["input"]]
    outputs = pipeline(
        prompts,
        num_return_sequences=1,
        temperature=1.0,
        top_p=0.9,
        max_new_tokens=150,
        do_sample=True,
        eos_token_id=pipeline.tokenizer.eos_token_id,
        pad_token_id=pipeline.tokenizer.pad_token_id,
    )
    return {"output": [output[0]["generated_text"] for output in outputs]}


def main():
    # Paths
    MODEL_PATH = "UICHEOL-HWANG/Dacon-contest-obfuscation-ko-gemma-7b"
    TEST_FILE = "../data/test.csv"
    OUTPUT_FILES = "../data/submission.csv"

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",        # 자동으로 GPU로 분배
        torch_dtype=torch.float16  # FP16 사용
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Initialize pipeline
    print("Initializing pipeline...")
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=8  # 배치 크기 설정
    )

    # Load test data into Hugging Face Dataset
    print("Loading test data...")
    test_df = pd.read_csv(TEST_FILE)
    test_dataset = Dataset.from_pandas(test_df)

    # Process data in batches
    print("Processing test data in batches...")
    processed_dataset = test_dataset.map(
        lambda batch: process_batch(batch, text_pipeline),  # 직접 참조
        batched=True,
        batch_size=8
    )

    # Save results to a CSV file
    print("Saving results to file...")
    processed_dataset.to_csv(OUTPUT_FILES, index=False, encoding="utf-8-sig")
    print(f"Results saved to {OUTPUT_FILES}")


if __name__ == "__main__":
    main()
