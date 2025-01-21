from datasets import Dataset
from sklearn.model_selection import train_test_split
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

    def preprocess_data(self, data):
        """
        Prepares the dataset by tokenizing input and output texts.

        Args:
            data (pd.DataFrame): The input dataset as a pandas DataFrame.

        Returns:
            tuple: Tokenized train and test datasets.
        """
        # Convert pandas DataFrame to Dataset
        dataset = Dataset.from_pandas(data)

        # Ensure input and output columns are present
        if "input" not in dataset.column_names or "output" not in dataset.column_names:
            raise ValueError("Dataset must contain 'input' and 'output' columns.")

        # Tokenize input and output
        def tokenize_function(examples):
            inputs = self.tokenizer(
                examples["input"],  # Tokenize the input column
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )
            outputs = self.tokenizer(
                examples["output"],  # Tokenize the output column
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )
            inputs["labels"] = outputs["input_ids"]  # Add output tokens as labels
            return inputs

        # Split dataset into train and test sets
        split_data = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_data["train"]
        test_dataset = split_data["test"]

        # Tokenize datasets
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
            fp16=True,
            no_cuda=True
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
