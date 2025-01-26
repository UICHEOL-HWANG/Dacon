import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch 

# 프롬프트 생성 함수
def create_prompt(input_text):
    """
    망가진 글자를 복원시키기 위한 프롬프트 생성.
    """
    return (
        "<start_of_turn> Your task is to transform the given obfuscated Korean review into a clear, correct, "
        "and natural-sounding Korean review that reflects its original meaning.\n"
        f"Input: {input_text}\n"
        "<end_of_turn>\n"
        "<start_of_turn>Assistant:\n"
        "Output(Korea Only):"
    )


# 반복된 문구 제거 함수
def remove_repeated_phrases(text):
    """
    반복된 문구를 제거하여 결과를 정제.
    """
    phrases = text.split()
    seen = set()
    result = []
    for phrase in phrases:
        if phrase not in seen:
            result.append(phrase)
            seen.add(phrase)
    return " ".join(result)


def main():
    # 경로 설정
    MODEL_PATH = "../models/gemma2-2b-it/merge/"
    TEST_FILE = "../data/test.csv"
    OUTPUT_FILE = "./submission.csv"

    # 모델 및 토크나이저 로드
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",        # GPU 자동 할당
        torch_dtype=torch.float16  # FP16 사용
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # 텍스트 생성 파이프라인 초기화
    print("Initializing text generation pipeline...")
    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    # 테스트 데이터 로드
    print("Loading test data...")
    test = pd.read_csv(TEST_FILE, encoding="utf-8-sig")

    restored_reviews = []

    # 데이터 처리
    print("Processing test data...")
    for index, row in test.iterrows():
        query = row['input']  # 입력 데이터
        prompt = create_prompt(query)

        print(f"Processing index {index}:")
        print(f"Input Query: {query}")
        print(f"Generated Prompt: {prompt}")

        # 텍스트 생성
        generated = text_gen_pipeline(
            prompt,
            num_return_sequences=1,
            temperature=0.2,
            top_p=0.9,
            max_new_tokens=len(query),
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        # 생성된 텍스트에서 결과 추출
        generated_text = generated[0]['generated_text']
        print(f"Generated Text: {generated_text}")

        # 'Output:' 이후 텍스트만 추출
        output_start = generated_text.find("Output:")
        if output_start != -1:
            result = generated_text[output_start + len("Output:"):].strip()
        else:
            result = generated_text.strip()

        # '<end_of_turn>' 이전 텍스트만 유지
        result = result.split("<end_of_turn>")[0].strip()

        # 반복된 문구 제거
        result = remove_repeated_phrases(result)
        print(f"Final Processed Output: {result}")

        # 결과 저장
        restored_reviews.append(result)
        print("-" * 50)  # 로그 구분선

    # 결과를 저장할 파일 생성
    print("Saving results...")
    submission = pd.DataFrame({"input": test['input'], "output": restored_reviews})
    submission.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
