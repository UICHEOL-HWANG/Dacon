from src.custom_dataset import CustomDataset
from src.model_manager import ModelManager
from src.training_manager import TrainingManager
import argparse
from src.split_data import split_data
import pandas as pd 

def main():
    parser = argparse.ArgumentParser(description="훈련용 자동 매개변수 지정")
    
    parser.add_argument('--learning_rate', type=float, default=3e-5, help="옵티마이저 하이퍼파리미터")
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size for training and validation')
    parser.add_argument('--train_data', type=str, default='../data/train.csv', help='Training dataset split')
    parser.add_argument('--max_length', type=int, default=128, help='max_length')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='output models')
    
    args = parser.parse_args() 
    
    controller = ModelManager()
    
    model = controller.model
    tokenizer = controller.tokenizer
    
    training_manager = TrainingManager(
        model=model,
        tokenizer=tokenizer,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    train, test = split_data(pd.read_csv(args.train_data))
    
    
    train_loader = CustomDataset(train, tokenizer=tokenizer, max_length=args.max_length)
    valid_loader = CustomDataset(test, tokenizer=tokenizer, max_length=args.max_length)
    training_manager.train(train_dataset=train_loader, 
                           eval_dataset=valid_loader
    )
    
    training_manager.save_model(args.output_dir)
    
if __name__ == "__main__":
    main()