from datasets import Dataset
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer
from peft import PeftModel
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

    def create_prompt(self, input_text, output_text=None):
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
        dataset = Dataset.from_pandas(data)

        if "input" not in dataset.column_names or "output" not in dataset.column_names:
            raise ValueError("Dataset must contain 'input' and 'output' columns.")

        def tokenize_function(examples):
            prompts = [
                self.create_prompt(input_text=examples["input"][i], output_text=examples["output"][i])
                for i in range(len(examples["input"]))
            ]
            tokenized = self.tokenizer(
                prompts,
                truncation=True,
                max_length=self.max_length,
                padding="longest",
                return_tensors="pt",
            )
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized

        split_data = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_data["train"]
        test_dataset = split_data["test"]

        train = train_dataset.map(tokenize_function, batched=True, num_proc=4)
        test = test_dataset.map(tokenize_function, batched=True, num_proc=4)

        return train, test

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

        # LoRA 어댑터 저장
        ADAPTER_MODEL = "lora_adapter_7b"
        trainer.model.save_pretrained(ADAPTER_MODEL)
        self.tokenizer.save_pretrained(ADAPTER_MODEL)
        print(f"LoRA Adapter model and tokenizer saved to {ADAPTER_MODEL}")

        # 파인튜닝 통합 모델 저장
        BASE_MODEL = "beomi/gemma-ko-7b"
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", torch_dtype=torch.float16)
        model = PeftModel.from_pretrained(model, ADAPTER_MODEL, device_map="auto", torch_dtype=torch.float16)
        model.save_pretrained("gemma_7b_finetuning")
        self.tokenizer.save_pretrained("gemma_7b_finetuning")
        print(f"Fine-tuned model and tokenizer saved to gemma_7b_finetuning")
