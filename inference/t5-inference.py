import pandas as pd 
from transformers import T5Tokenizer, T5ForConditionalGeneration

def main():
    MODEL_PATH = "../models/outputs"
    FILES = "../data/test.csv"
    OUTPUT_FILES = "../data/submission.csv"

    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)

    test = pd.read_csv(FILES)

    restored_reviews = []

    # 테스트 데이터에서 각 문장에 대해 텍스트 생성
    for index, row in test.iterrows():
        query = row["input"]
        
        # 프롬프트 생성
        prompt = f"restore: {query}"
        
        # 입력 토큰화
        input_ids = tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True).input_ids

        # 모델 추론
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

        # 생성된 텍스트 디코딩
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 로그 출력: 원본 입력과 생성된 출력
        print(f"Processing Row {index + 1}/{len(test)}")
        print(f"Input: {query}")
        print(f"Output: {generated_text}\n")

        # 결과 저장
        restored_reviews.append(generated_text.strip())
        
    # 결과를 test 데이터프레임에 추가
    test["output"] = restored_reviews

    # 결과를 새로운 CSV 파일로 저장
    test.to_csv(OUTPUT_FILES, index=False, encoding="utf-8-sig")

    print(f"결과가 {OUTPUT_FILES}에 저장되었습니다.")

# 직접 실행될 때만 main() 호출
if __name__ == "__main__":
    main()
