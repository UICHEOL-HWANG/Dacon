from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch

class ModelManager:
    
    def __init__(self):
        # BitsAndBytesConfig 설정
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',            # NF4로 설정
            bnb_4bit_use_double_quant=True,       # Double Quantization 활성화
            bnb_4bit_compute_dtype=torch.bfloat16 # 계산 데이터 타입 (bfloat16)
        )

        # 모델 경로
        self.model_path = "google/gemma-2-2b-it"

        # 4-bit 양자화 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            quantization_config=bnb_config,
        )

        # LoRA 설정
        lora_config = LoraConfig(
            r=8,                                 # LoRA rank
            lora_alpha=32,                        # LoRA scaling factor
            target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj", 
                "gate_proj", "down_proj", "up_proj"
            ],                                    # 수정할 레이어
            lora_dropout=0.1,                     # 드롭아웃 비율
            bias="none",                          # Bias 처리 방법
            task_type="CAUSAL_LM"                 # 언어 모델 타입
        )
        # 모델에 LoRA 적용
        self.model = get_peft_model(self.model, lora_config)

        # Tokenizer 로드 및 설정
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
