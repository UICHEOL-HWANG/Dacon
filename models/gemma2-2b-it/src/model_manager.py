from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 

class ModelManger:
    
    def __init__(self):
        self.model_path = "google/gemma-2-2b-it"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path
        )
        
        self.tokenizer.pad_token=self.tokenizer.eos_token
        self.tokenizer.padding_side="right"