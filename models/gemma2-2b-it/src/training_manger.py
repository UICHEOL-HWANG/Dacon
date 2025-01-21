from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
import torch


class TrainingManager:
    def __init__(self, model, tokenizer, train_file, output_dir, max_length=512, batch_size=8, learning_rate=5e-5, num_epochs=3):
        self.model = model
        self.tokenizer = tokenizer
        self.train_file = train_file
        self.output_dir = output_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

    @staticmethod
    def create_prompt(input_text, output_text=None):
        """
        주어진 입력과 출력 텍스트를 기반으로 프롬프트를 생성합니다.
        output_text가 None이면 테스트 시 사용됩니다.
        """
        if output_text:
            return (
                "<start_of_turn> Your task is to transform the given obfuscated Korean review into a clear, correct, "
                "and natural-sounding Korean review that reflects its original meaning.\n"
                f"Input: {input_text}\n"
                "<end_of_turn>\n"
                "<start_of_turn>Assistant:\n"
                f"Output: {output_text}"
            )
        else:
            return (
                "<start_of_turn> Your task is to transform the given obfuscated Korean review into a clear, correct, "
                "and natural-sounding Korean review that reflects its original meaning.\n"
                f"Input: {input_text}\n"
                "<end_of_turn>\n"
                "<start_of_turn>Assistant:\n"
                "Output:"
            )

    def preprocess_data(self, data):
        """
        Prepares the dataset by applying the prompt template and tokenizing input and output texts.

        Args:
            data (pd.DataFrame): The input dataset as a pandas DataFrame.

        Returns:
            tuple: Tokenized train and test datasets.
        """
        # 샘플링 (다양성을 위해 비율로 설정)
        data = data.sample(frac=0.1, random_state=42).reset_index(drop=True)

        # Dataset 변환
        dataset = Dataset.from_pandas(data)

        # 입력과 출력 열 존재 여부 확인
        if "input" not in dataset.column_names or "output" not in dataset.column_names:
            raise ValueError("Dataset must contain 'input' and 'output' columns.")

        # 프롬프트 생성 및 토크나이징 함수
        def tokenize_function(examples):
            prompts = [
                self.create_prompt(input_text=examples["input"][i], output_text=examples["output"][i])
                for i in range(len(examples["input"]))
            ]
            tokenized = self.tokenizer(
                prompts,
                truncation=True,
                max_length=self.max_length,
                padding="longest",  # 동적으로 패딩
                return_tensors="pt",
            )
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized

        # Train/Test Split
        split_data = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_data["train"]
        test_dataset = split_data["test"]

        # 토크나이징 수행
        train = train_dataset.map(tokenize_function, batched=True)
        test = test_dataset.map(tokenize_function, batched=True)

        return train, test

    def train(self, train_data, test_data):
        """
        Trains the model using Hugging Face SFTTrainer.

        Args:
            train_data: Tokenized training dataset.
            test_data: Tokenized test dataset.
        """
        # Set training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.batch_size,
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

        # Initialize SFTTrainer
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            tokenizer=self.tokenizer,
        )

        # Start training
        trainer.train()

        # Save the model and tokenizer
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"Model fine-tuned and saved to {self.output_dir}")
