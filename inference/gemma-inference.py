from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
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

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Initialize pipeline
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    # Load test data
    test = pd.read_csv(TEST_FILE)
    test_subset = test.iloc[:3].copy()  # Select only the first 3 samples
    restored_reviews = []

    # Process test data
    for index, row in test_subset.iterrows():
        query = row["input"]

        # Create prompt for the current test input
        prompt = create_prompt(query)

        # Generate output from the model
        outputs = text_pipeline(
            prompt,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=150,
            do_sample=True,
        )

        # Extract the generated text
        generated_text = outputs[0]["generated_text"]

        # Remove "Output:" and trailing whitespaces
        output_start = generated_text.find("Output:")
        if output_start != -1:
            result = generated_text[output_start + len("Output:"):].strip()
        else:
            result = generated_text.strip()

        # Debugging output
        print(f"Processing Row {index + 1}/{len(test_subset)}")
        print(f"Input: {query}")
        print(f"Output: {result}\n")

        # Append result
        restored_reviews.append(result)

    # Save results to a CSV file
    test_subset["output"] = restored_reviews
    test_subset.to_csv(OUTPUT_FILES, index=False, encoding="utf-8-sig")
    print(f"Results saved to {OUTPUT_FILES}")


if __name__ == "__main__":
    main()
