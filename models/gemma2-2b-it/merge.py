from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch 

BASE_MODEL_PATH = "google/gemma-2-2b"  # 본 모델 경로
ADAPTER_MODEL_PATH = "UICHEOL-HWANG/Dacon-contest-obfuscation-gemma2-2b"  # 어댑터 저장된 경로
OUTPUT_DIR = "./merge"  # 합쳐진 모델 저장 경로

# 본 모델 로드
model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
                torch_dtype=torch.float16,
                    device_map="auto"
                    )

# 어댑터 가중치 로드
model = PeftModel.from_pretrained(
            model,
                ADAPTER_MODEL_PATH,
                    torch_dtype=torch.float16,
                        device_map="auto"
                        )

model = model.merge_and_unload()

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

# 병합된 모델 저장
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Merged model and tokenizer saved to {OUTPUT_DIR}")