import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch

def main():
    # Hugging Face 로그인
    login(token="token")  # 실제 토큰 사용
    
    # 데이터 경로 설정
    file_path = "C:/Users/user/Desktop/open/data/test.csv"
    train_path = "C:/Users/user/Desktop/open/data/train.csv"

    # 데이터 로드
    train = pd.read_csv(train_path)
    test = pd.read_csv(file_path)
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 학습 데이터에서 샘플 5개 추출
    samples = [
        f"Input: {row['input']} → Output: {row['output']}"
        for _, row in train.iloc[:5].iterrows()
    ]

    # 모델과 토크나이저 로드
    main_path = "google/gemma-2-2b-it"
    model = AutoModelForCausalLM.from_pretrained(main_path).to(device)  # 모델을 GPU로 이동
    tokenizer = AutoTokenizer.from_pretrained(main_path)

    # 테스트 데이터 처리
    for index, row in test[:5].iterrows():
        query = row['input']

        # 프롬프트 생성
        system_prompt = f"You are a helpful assistant specializing in restoring obfuscated Korean reviews. \
        Your task is to transform the given obfuscated Korean review into a clear, correct, \
        and natural-sounding Korean review that reflects its original meaning. \
        Below are examples of obfuscated Korean reviews and their restored forms:\n\n" + "\n".join(samples) + f"\n\nInput: {query}\nOutput:"
        
        # 입력 토큰화 및 GPU로 이동
        input_ids = tokenizer(
            system_prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).input_ids.to(device)

        # 모델 추론
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=128,  # 생성 토큰의 최대 길이
            temperature=0.7,  # 다양성 제어
            top_p=0.9,  # 확률 컷오프
            do_sample=True  # 샘플링 활성화
        )

        # 생성된 텍스트 디코딩
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 결과 추출
        output_start = generated_text.find("Output:")
        if output_start != -1:
            result = generated_text[output_start + len("Output:"):].strip()
        else:
            result = generated_text.strip()

        # 결과 출력
        print(f"Processed {index + 1}/{len(test)}")
        print(f"Input: {query}")
        print(f"Output: {result}\n")

# 이 구문을 통해 스크립트가 직접 실행될 때만 main() 호출
if __name__ == "__main__":
    main()
