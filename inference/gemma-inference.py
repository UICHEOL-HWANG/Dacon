import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import pandas as pd
import torch 

def create_prompt(input_text):
    """
    망가진 글자를 복원 시키기 위한 프롬프트 
    """
    return (
        "<start_of_turn> Your task is to transform the given obfuscated Korean review into a clear, correct, "
        "and natural-sounding Korean review that reflects its original meaning.\n"
        f"Input: {input_text}\n"
        "<end_of_turn>\n"
        "<start_of_turn>Assistant:\n"
        "Output:"
    )


def clean_generated_text(generated_text):
    """
    regex를 이용한 문자열 클렌징
    """
    # Find 'Output:' and get everything after it
    output_start = generated_text.find("Output:")
    if output_start != -1:
        result = generated_text[output_start + len("Output:"):].strip()
    else:
        result = generated_text.strip()

    # Remove unwanted tokens like <end_of_turn>, <start_of_turn>, etc.
    result = re.sub(r"<.*?>", "", result).strip()
    return result


def main():
    # Paths
    MODEL_PATH = "UICHEOL-HWANG/Dacon-contest-obfuscation-ko-gemma-7b"
    TEST_FILE = "../data/test.csv"
    OUTPUT_FILES = "../data/submission.csv"
    
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                       # 4비트 양자화 활성화
    bnb_4bit_quant_type="nf4",               # NF4 양자화 방식
    bnb_4bit_use_double_quant=True,          # Double Quantization 활성화
    bnb_4bit_compute_dtype=torch.bfloat16    # 계산 데이터 타입 (bfloat16 사용)
    )


    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Initialize pipeline
    print("Initializing pipeline...")
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    # Load test data
    print("Loading test data...")
    test = pd.read_csv(TEST_FILE)
    restored_reviews = []

    # Process test data
    print("Processing test data...")
    for index, row in test.iterrows():
        query = row["input"]

        # Create prompt for the current test input
        prompt = create_prompt(query)

        # Generate output from the model
        outputs = text_pipeline(
            prompt,
            num_return_sequences=1,
            temperature=1.0,
            top_p=0.9,
            max_new_tokens=150,
            do_sample=True,
        )

        # Extract the generated text
        generated_text = outputs[0]["generated_text"]

        # Clean the output text
        result = clean_generated_text(generated_text)

        # Debugging output (optional)
        print(f"Processing Row {index + 1}/{len(test)}")
        print(f"Input: {query}")
        print(f"Output: {result}\n")

        # Append result
        restored_reviews.append(result)

    # Save results to a CSV file
    print("Saving results to file...")
    test["output"] = restored_reviews
    test.to_csv(OUTPUT_FILES, index=False, encoding="utf-8-sig")
    print(f"Results saved to {OUTPUT_FILES}")


if __name__ == "__main__":
    main()
