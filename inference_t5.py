"""
author: Difei Tang
email: DIT18@pitt.edu
MeLoDy Lab, University of Pittsburgh

Script for inference using trained T5 model
"""

import argparse
from pathlib import Path
from T5_trainer import T5Trainer
import json

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
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Perform BLTL inference using trained T5 model')
    parser.add_argument('--model_path', type=str, 
                      help='Path to the trained model')
    parser.add_argument('--input_text', type=str,
                      help='Natural language input to convert to BLTL')
    parser.add_argument('--data_path', type=str, default='data/total_data_Apr09.jsonl',
                      help='Path to the data directory')
    parser.add_argument('--output_dir', type=str, default='output/evaluated_results/',
                      help='Directory to save the evaluated results (default: output/evaluated_results/)')

    args = parser.parse_args()

    # Load data
    if args.data_path:
        data_path = args.data_path
        if not validate_jsonl(data_path):
            print("Invalid JSONL file. Exiting.")
            return
        
        print(f"Loading data from {data_path}...")
        dataset = load_dataset('json', data_files=data_path, split='train')
        
        print(f"Dataset: {len(dataset)}")
    elif args.input_text:
        input_text = args.input_text
    else:
        print("No input text or data path provided.")
        return

    # Initialize the trainer with the same configuration used during training
    trainer = T5Trainer(
        model_checkpoint='t5-large',
        max_input_length=2048,
        max_target_length=2048,
        output_dir=args.output_dir
    )

    # Make prediction
    try:
        if args.data_path:
            trainer.evaluate(test_dataset=dataset, model_path=args.model_path)
        else:
            trainer.predict(input_text=input_text, model_path=args.model_path)
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")

if __name__ == "__main__":
    main()