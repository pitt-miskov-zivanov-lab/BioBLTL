"""
author: Difei Tang
email: DIT18@pitt.edu
MeLoDy Lab, University of Pittsburgh
"""

import argparse
from pathlib import Path
from T5_trainer import T5Trainer
import json
import os
from datasets import load_dataset

def validate_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            try:
                json.loads(line.strip())
            except json.JSONDecodeError:
                print(f"Error in line {i}: {line}")
                return False
    return True

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train T5 model for BLTL property generation')
    parser.add_argument('--model', type=str, default='t5-small',
                      help='Model checkpoint to use (default: t5-small)')
    parser.add_argument('--trained_model_path', type=str, default=None,
                      help='Path to the trained model (default: None)')
    parser.add_argument('--data_dir', type=str, default='data/nl_bltl',
                      help='Path to the data directory')
    parser.add_argument('--data_basename', type=str, default='tcell_bltl')
    parser.add_argument('--output_dir', type=str, default='output/trained_models/',
                      help='Directory to save the trained model (default: output/trained_models/)')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size for training (default: 8)')
    parser.add_argument('--num_epochs', type=int, default=2,
                      help='Number of training epochs (default: 10)')
    parser.add_argument('--eval_steps', type=int, default=500,
                      help='Number of evaluation steps (default: 500)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Learning rate for training (default: 1e-4)')
    
    args = parser.parse_args()

    # Load data
    data_dir = args.data_dir

    if os.path.exists(f'{data_dir}/{args.data_basename}_eval.json'):
        dataset = load_dataset('json', data_files={
            'train': f'{data_dir}/{args.data_basename}_train.json',
            'test': f'{data_dir}/{args.data_basename}_test.json',
            'validation': f'{data_dir}/{args.data_basename}_eval.json'
        })
    else:
        dataset = load_dataset('json', data_files={
            'train': f'{data_dir}/{args.data_basename}_train.json',
            'test': f'{data_dir}/{args.data_basename}_test.json'
        })

    train_dataset = dataset['train']
    eval_dataset = dataset['validation'] if 'validation' in dataset else None
    test_dataset = dataset['test']

    #Initialize trainer
    model_checkpoint = args.model
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    output_dir = args.output_dir
    trained_model_path = args.trained_model_path

    if trained_model_path:
        print(f"Loading trained model from {trained_model_path}...")

    trainer = T5Trainer(
        model_checkpoint=model_checkpoint,
        trained_model_path=trained_model_path,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        output_dir=output_dir
    )
    
    # Train model
    print("Starting training...")
    trainer.train(train_dataset, eval_dataset, eval_steps=args.eval_steps)
    
    print("Training completed!")

    # Evaluate model
    print("Starting evaluation...")
    trainer.evaluate(test_dataset)
    print("Evaluation completed!")

if __name__ == "__main__":
    main() 