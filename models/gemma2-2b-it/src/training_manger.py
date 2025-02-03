from datasets import Dataset
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer
from peft import PeftModel
import torch
from functools import partial

class TrainingManager:
    def __init__(self, model, tokenizer, train_file, output_dir, max_length=512, batch_size=8, learning_rate=5e-5, num_epochs=3):
        self.model = model
        self.tokenizer = tokenizer
        self.train_file = train_file  # 예: CSV 파일 경로 등 (데이터 로딩 시 활용)
        self.output_dir = output_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

    @staticmethod
    def create_prompt(input_text, output_text):
        prompt = (
            "<start_of_turn> Your task is to transform the given obfuscated Korean review into a clear, correct, and natural-sounding Korean review that reflects its original meaning.\n"
            f"Input: {input_text}\n"
            "<end_of_turn>\n"
            "<start_of_turn>Assistant:\n"
            f"Output: {output_text}"
        )
        return prompt

    @staticmethod
    def format_chat_template(row, tokenizer, max_length):
        # row는 딕셔너리 형태로 "input"과 "output" 키를 가져야 합니다.
        prompt = TrainingManager.create_prompt(row["input"], row["output"])
        # encode()를 사용하면 토큰 ID 리스트를 반환합니다.
        tokens = tokenizer.encode(prompt, truncation=True, max_length=max_length)
        row["input_ids"] = tokens
        return row

    def preprocess_data(self, data):
        """
        data: pandas.DataFrame 혹은 datasets.Dataset
        데이터셋에 "input"과 "output" 컬럼이 있다고 가정합니다.
        """
        # 만약 data가 pandas DataFrame이면 Dataset으로 변환
        if not isinstance(data, Dataset):
            data = Dataset.from_pandas(data)

        # 컬럼 확인
        if "input" not in data.column_names or "output" not in data.column_names:
            raise ValueError("Dataset must contain 'input' and 'output' columns.")

        # format_chat_template를 적용 (batched=False 사용)
        # 내부 함수가 모듈 최상위에 있으므로 멀티프로세싱 시 피클링 문제를 최소화
        tokenize_fn = partial(TrainingManager.format_chat_template, tokenizer=self.tokenizer, max_length=self.max_length)
        processed_dataset = data.map(tokenize_fn, batched=False, num_proc=4)

        # train/test split (예: 90% train, 10% test)
        split_data = processed_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_data["train"]
        test_dataset = split_data["test"]

        return train_dataset, test_dataset

    def train(self, train_data, test_data):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=4,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_epochs,
            evaluation_strategy="steps",
            eval_steps=100,
            save_steps=200,
            save_total_limit=2,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=50,
            fp16=torch.cuda.is_available(),
        )

        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            tokenizer=self.tokenizer,
        )

        trainer.train()

        trainer.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"LoRA Adapter model and tokenizer saved to {self.output_dir}")