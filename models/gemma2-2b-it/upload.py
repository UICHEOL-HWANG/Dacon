from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = "./merge"

# 병합된 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    OUTPUT_DIR,
    device_map={"": 0}  # GPU 0으로 매핑
)

# 병합된 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)

# 모델 업로드
model.push_to_hub("UICHEOL-HWANG/Dacon-contest-obfuscation-gemma2-2b")
tokenizer.push_to_hub("UICHEOL-HWANG/Dacon-contest-obfuscation-gemma2-2b")

print("Model and tokenizer pushed to Hugging Face Hub.")
