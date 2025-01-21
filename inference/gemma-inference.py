import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def remove_repeated_phrases(text):
    phrases = text.split(" ")
    seen = set()
    result = []
    for phrase in phrases:
        if phrase not in seen:
            result.append(phrase)
            seen.add(phrase)
    return " ".join(result)

def main():
    MODEL_PATH = "UICHEOL-HWANG/Dacon-contest-obfuscation-gemma2-2b"
    TRAIN_FILE = "../data/train.csv"
    TEST_FILE = "../data/test.csv"
    OUTPUT_FILES = "../data/submission.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # 학습 데이터 로드 및 샘플 생성
    train = pd.read_csv(TRAIN_FILE)
    samples = [
        f"Input: {train['input'][i]} → Output: {train['output'][i]}"
        for i in range(min(5, len(train)))
    ]
    system_prompt = "\n".join(samples)

    # 테스트 데이터 로드
    test = pd.read_csv(TEST_FILE)
    restored_reviews = []

    # 테스트 데이터 처리
    for index, row in test.iterrows():
        query = row["input"]

        # 프롬프트 생성
        prompt = (
            f"{system_prompt}\n\n"
            f"Input: {query}\n"
            "Output:"
        )

        # 입력 토큰화 및 GPU로 이동
        input_ids = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).input_ids.to(device)

        # 모델 추론
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.85,
            do_sample=True,
        )

        # 생성된 텍스트 디코딩
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 중복된 구문 제거
        result = remove_repeated_phrases(generated_text)

        # 디버깅 출력
        print(f"Processing Row {index + 1}/{len(test)}")
        print(f"Input: {query}")
        print(f"Output: {result}\n")

        restored_reviews.append(result.strip())

    # 결과 저장
    test["output"] = restored_reviews
    test.to_csv(OUTPUT_FILES, index=False, encoding="utf-8-sig")
    print(f"Results saved to {OUTPUT_FILES}")

if __name__ == "__main__":
    main()
