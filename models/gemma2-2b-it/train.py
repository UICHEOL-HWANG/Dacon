from src.training_manger import TrainingManager
from src.model_manager import ModelManager
import argparse

import pandas as pd 

def main():
    
    parser = argparse.ArgumentParser(description="Fine-tune a model using SFTTrainer.")
    
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training dataset (CSV format).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for tokenization.")
    parser.add_argument("--batch_size", type=int, default=10, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for training.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    
    args = parser.parse_args()
    
    
    model_manager = ModelManager()
    
    model = model_manager.model 
    tokenizer = model_manager.tokenizer
    
    data = pd.read_csv(args.train_file)
    
    training_manager = TrainingManager(
        model=model,
        tokenizer=tokenizer,
        train_file=args.train_file,
        output_dir=args.output_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
    )
    
    train_data, test_data = training_manager.preprocess_data(data)
    
    training_manager.train(train_data, test_data)

if __name__ == "__main__":
    main()