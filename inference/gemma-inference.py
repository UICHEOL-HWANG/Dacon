import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def create_prompt(input_text, output_text=None):
    """
    Creates a prompt for the model to generate outputs.
    """
    if output_text:
        return (
            f"Transform the obfuscated Korean review into its natural form:\n"
            f"Input: {input_text}\n"
            f"Output: {output_text}"
        )
    else:
        return (
            f"Transform the obfuscated Korean review into its natural form:\n"
            f"Input: {input_text}\n"
            "Output:"
        )


def remove_repeated_phrases(text):
    """
    Removes repeated phrases from the generated text.
    """
    phrases = text.split(" ")
    seen = set()
    result = []
    for phrase in phrases:
        if phrase not in seen:
            result.append(phrase)
            seen.add(phrase)
    return " ".join(result)


def main():
    # Paths
    MODEL_PATH = "UICHEOL-HWANG/Dacon-contest-obfuscation-gemma2-2b"
    TRAIN_FILE = "../data/train.csv"
    TEST_FILE = "../data/test.csv"
    OUTPUT_FILES = "../data/submission.csv"

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Load training data and create system prompt
    train = pd.read_csv(TRAIN_FILE)
    samples = [
        create_prompt(input_text=train['input'][i], output_text=train['output'][i])
        for i in range(min(3, len(train)))  # Use top 3 samples
    ]
    system_prompt = "\n\n".join(samples)

    # Load test data
    test = pd.read_csv(TEST_FILE)
    test_subset = test.iloc[:3].copy()  # 테스트 데이터의 상위 3개를 복사
    restored_reviews = []

    # Process test data
    for index, row in test_subset.iterrows():
        query = row["input"]

        # Create prompt for the current test input
        prompt = f"{system_prompt}\n\n{create_prompt(input_text=query)}"

        # Tokenize input and move to device
        input_ids = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).input_ids.to(device)

        # Generate output from the model
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.85,
            do_sample=True,
        )

        # Decode and process the generated output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = remove_repeated_phrases(generated_text)

        # Debugging output
        print(f"Processing Row {index + 1}/{len(test_subset)}")
        print(f"Input: {query}")
        print(f"Output: {result}\n")

        # Append result
        restored_reviews.append(result.strip())

    # Save results to a CSV file
    test_subset["output"] = restored_reviews
    test_subset.to_csv(OUTPUT_FILES, index=False, encoding="utf-8-sig")
    print(f"Results saved to {OUTPUT_FILES}")


if __name__ == "__main__":
    main()
