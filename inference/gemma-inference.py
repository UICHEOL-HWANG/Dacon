from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import pandas as pd


def create_prompt(input_text):
    """
    Creates a simple prompt for the model to generate outputs.
    """
    return (
        "<start_of_turn> Your task is to transform the given obfuscated Korean review into a clear, correct, "
        "and natural-sounding Korean review that reflects its original meaning.\n"
        f"Input: {input_text}\n"
        "<end_of_turn>\n"
        "<start_of_turn>Assistant:\n"
        "Output:"
    )


def main():
    # Paths
    MODEL_PATH = "UICHEOL-HWANG/Dacon-contest-obfuscation-ko-gemma-7b"
    TEST_FILE = "../data/test.csv"
    OUTPUT_FILES = "../data/submission.csv"


    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True  # Stability improvement for INT8
    )

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        quantization_config=quantization_config
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


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

        # Validate input
        if not query.strip():
            query = "Default prompt for empty input."

        # Create prompt for the current test input
        prompt = create_prompt(query)

        # Generate output from the model
        try:
            outputs = text_pipeline(
                prompt,
                num_return_sequences=1,
                temperature=1.0,
                top_p=0.9,
                max_new_tokens=150,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

            # Extract the generated text
            generated_text = outputs[0]["generated_text"]

            # Remove "Output:" and trailing whitespaces
            output_start = generated_text.find("Output:")
            if output_start != -1:
                result = generated_text[output_start + len("Output:"):].strip()
            else:
                result = generated_text.strip()

            # Append result
            restored_reviews.append(result)

        except Exception as e:
            print(f"Error processing row {index}: {e}")
            restored_reviews.append("Error")

    # Save results to a CSV file
    print("Saving results to file...")
    test["output"] = restored_reviews
    test.to_csv(OUTPUT_FILES, index=False, encoding="utf-8-sig")
    print(f"Results saved to {OUTPUT_FILES}")


if __name__ == "__main__":
    main()
