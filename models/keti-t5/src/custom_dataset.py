from torch.utils.data import Dataset
import pandas as pd


class CustomDataset(Dataset):
    
    def __init__(self, data, tokenizer, max_length=128):
        self.data = pd.read_csv(data)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # 데이터 행 가져오기
        row = self.data.iloc[index]
        
        # 망가진 문장 및 올바른 문장
        inputs_text = row['input']
        outputs_text = row['output']
        
        # 토크나이징
        inputs = self.tokenizer(
            inputs_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        labels = self.tokenizer(
            outputs_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels["input_ids"].squeeze(0)
        }
