from transformers import T5Tokenizer, T5ForConditionalGeneration

import os


class ModelManager:
    def __init__(self):
        self.model_path = "KETI-AIR/ke-t5-base-ko"
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_path)
    
        