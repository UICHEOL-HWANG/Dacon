from transformers import Trainer, TrainingArguments
import torch 
from torch.utils.data import DataLoader
from tqdm import tqdm 
import numpy as np 


class TrainingManager:
    
    def __init__(self, model, tokenizer, learning_rate=5e-5, epochs=5, batch_size=8, device=None):
        
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        # Device 설정 (자동 감지 또는 수동 설정)
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def train(self, train_dataset):
        
        training_args = TrainingArguments(
            output_dir="./outputs",
            eval_strategy="no",
            num_train_epochs=self.epochs,
            gradient_accumulation_steps=2,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            logging_strategy="steps",
            save_strategy="epoch",
            save_steps=250,
            logging_dir='./logs',
            logging_steps=100,
            dataloader_num_workers=0,
        )

        # Trainer 객체 생성
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset
        )

        # 학습 실행
        trainer.train()

    def save_model(self, output_dir="./model"):
        print(f"Saving model to {output_dir}...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print("Model and tokenizer saved successfully.")
