from transformers import Trainer, TrainingArguments
import torch 
from torch.utils.data import DataLoader
from tqdm import tqdm 
import numpy as np 
from sklearn.metrics import f1_score, precision_score, recall_score



def compute_char_level_f1(pred):
    """
    문자 단위 F1 Score 계산 (Precision, Recall 포함)
    pred: (predictions, labels) 튜플로 반환됨
    """
    preds, labels = pred
    preds = preds.argmax(axis=-1)  # 예측값 (Logits에서 argmax)
    preds = preds.squeeze().tolist()
    labels = labels.squeeze().tolist()

    # 정확히 일치한 문자 수 (num_same)
    num_same = sum([1 for p, l in zip(preds, labels) if p == l])

    # Precision, Recall, F1 계산
    precision = num_same / len(preds) if len(preds) > 0 else 0
    recall = num_same / len(labels) if len(labels) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

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
        
    def train(self, train_dataset, eval_dataset):
        training_args = TrainingArguments(
            output_dir="./outputs",
            eval_strategy="epoch", 
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            save_steps=250,
            logging_dir='./logs',
            logging_steps=100,
        )

        # Trainer 객체 생성
        trainer = Trainer(
            model=self.model,                           
            args=training_args,                   
            train_dataset=train_dataset,             
            eval_dataset=eval_dataset,
            compute_metrics=compute_char_level_f1     
        )

        # 학습 실행
        trainer.train()

    def save_model(self, output_dir="./model"):
        print(f"Saving model to {output_dir}...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print("Model and tokenizer saved successfully.")