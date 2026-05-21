"""
author: Difei Tang
email: DIT18@pitt.edu
MeLoDy Lab, University of Pittsburgh
"""

import torch
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset
from typing import Dict, List, Optional, Tuple
import json
import random
import csv
import os
import logging 
import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

from bltl_evaluator import BLTLEvaluator
from evaluate_llm import evaluate_results

class T5Trainer:
    def __init__(
        self,
        model_checkpoint: str = "t5-large",
        trained_model_path: str = None,
        max_input_length: int = 2048,
        max_target_length: int = 2048,
        batch_size: int = 8,
        num_epochs: int = 10,
        learning_rate: float = 1e-4,
        output_dir: str = 'output/trained_models/'
    ):
        self.model_checkpoint = model_checkpoint
        self.trained_model_path = trained_model_path
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.output_dir = Path(output_dir)
        self.num_epochs = num_epochs 
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # instead of string matching, we use the evaluator to check the generated BLTL 
        self.evaluator = BLTLEvaluator()
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        
        # Set prefix based on model
        self.prefix = "Transform the following sentence into Bounded Linear Temporal logic (BLTL):" if model_checkpoint in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"] else ""
        
        # Initialize model
        if self.trained_model_path:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.trained_model_path)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        
        # Initialize data collator
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
    
        self.setup_logger() 

    def setup_logger(self):
        """Set up logging configuration"""
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create a unique log file name with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"t5_training_{timestamp}.log"
        
        # Configure logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()  # Also print to console
            ]
        )
        self.logger = logging.getLogger(__name__)

    def preprocess_function(self, examples: Dict) -> Dict:
        """
        Preprocess the examples for model input.
        
        Args:
            examples (Dict): Input examples
            
        Returns:
            Dict: Preprocessed examples
        """

        if 'instruction' in examples and examples['instruction']:
            inputs = [inst + " " + doc for inst, doc in zip(examples["instruction"], examples["input"])]
        else:
            inputs = [self.prefix + " " + doc for doc in examples["input"]]

        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_input_length,
            truncation=True
        )
        
        # setup the tokenizer for labels 
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples["output"],
                max_length=self.max_target_length,
                truncation=True
            )
         
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["input"] = examples["input"]
        model_inputs["output"] = examples["output"]

        return model_inputs
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        # print(predictions)
        # print(labels)
        # Replace -100 in the labels as we can't decode them.
        predictions = np.where(predictions != -100, predictions, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        count = 0

        for i in range(len(decoded_preds)):
            pred_tokens = nltk.sent_tokenize(decoded_preds[i].strip())
            label_tokens = nltk.sent_tokenize(decoded_labels[i].strip())

            pred = decoded_preds[i].strip()
            label = decoded_labels[i].strip()
            if pred_tokens == label_tokens:
                count += 1
            else:
                evaluated_result = self.evaluator.evaluate_bltl(pred, label)
                if not evaluated_result: 
                    pass 
                elif evaluated_result['equivalent']:
                    count += 1

            self.logger.info(f"Pred: {pred}")
            self.logger.info(f"Label: {label}")

        return {'top-1 accuracy': round(count / len(decoded_preds), 6)}
    
    def train(self, train_dataset: Dataset, eval_dataset: Dataset = None, eval_steps: int = 500) -> None:
        """
        Train the model.
        
        Args:
            train_dataset (Dataset): Training dataset
            test_dataset (Dataset): Test dataset
        """

        import torch
        torch.cuda.empty_cache()
        # Tokenize datasets
        tokenized_train = train_dataset.map(self.preprocess_function, batched=True)
        tokenized_eval = eval_dataset.map(self.preprocess_function, batched=True) if eval_dataset else None
        
        # Get current timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.model_name = (
            self.model_checkpoint.split("/")[-1] + 
            f"-epoch{self.num_epochs}-test-train" +
            f"-{timestamp}"
        )
        
        # prepare the training arguments
        base_args = {
            "output_dir": str(self.output_dir / self.model_name),
            "learning_rate": self.learning_rate,
            "per_device_train_batch_size": self.batch_size,
            "per_device_eval_batch_size": self.batch_size,
            "weight_decay": 0.01,
            "seed": 1203,
            "save_total_limit": 1,
            "num_train_epochs": self.num_epochs,
            "predict_with_generate": True,
            "fp16": False,
            "generation_max_length": self.max_target_length,
            "save_strategy": "epoch",
        }

        if eval_dataset:
            base_args.update({
                "eval_strategy": "epoch",
                "eval_steps": eval_steps,
                "metric_for_best_model": "eval_loss",
                "load_best_model_at_end": True
            })

        args = Seq2SeqTrainingArguments(**base_args)

        trainer_args = {
            "model": self.model,
            "args": args,
            "train_dataset": tokenized_train,
            "data_collator": self.data_collator,
            "tokenizer": self.tokenizer,
            "compute_metrics": self.compute_metrics
        }

        if eval_dataset:
            trainer_args["eval_dataset"] = tokenized_eval

        # Initialize trainer
        trainer = Seq2SeqTrainer(**trainer_args)

        # Train the model
        trainer.train()
        
    def evaluate(self, test_dataset: Dataset, model_path: Optional[str] = None) -> None:
        """
        Evaluate the model on the test dataset.
        """

        # Evaluate and Save results
        tokenized_test = test_dataset.map(self.preprocess_function, batched=True)

        if model_path:
            self.model_name = model_path.split("/")[-1] + f"-inference"

        self.save_results(self.model_name, tokenized_test, model_path)
    
    def save_results(self, model_name: str, test_dataset: Dataset, model_path: Optional[str] = None) -> None:
        """
        Save model evaluation results.
        
        Args:
            model_name (str): Name of the model
            test_dataset (Dataset): Test dataset
            model_path (Optional[str]): Path to the trained model. If None, uses currently loaded model
        """

        if model_path:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        self.model.to(device)

        total_samples = len(test_dataset)

        output_list = []
        with open(self.output_dir / f'generated_predictions.jsonl', 'w') as f:
            for i, data in tqdm(enumerate(test_dataset), total=total_samples, desc="Evaluating"):
                inputs = [data['instruction'] + " " + data['input']]

                inputs = self.tokenizer(
                    inputs,
                    max_length=self.max_input_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(device)
                
                # FIXME: max_length might need to increase 
                output = self.model.generate(
                    **inputs,
                    num_beams=8,
                    do_sample=True,
                    min_length=1,
                    max_length=2048
                )

                decoded_output = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
                predicted_title = decoded_output.strip()

                prediction = {
                    'prompt': data['instruction'] + " " + data['input'],
                    'predict': predicted_title,
                    'label': data['output']
                }

                output_list.append(prediction)
                f.write(json.dumps(prediction, ensure_ascii=False) + '\n')

        evaluate_results(output_list, self.output_dir)

    def predict(self, input_text: str, model_path: Optional[str] = None, instruction: str = None) -> str:
        """
        Perform inference on new input text using the trained model.
        
        Args:
            input_text (str): The natural language input text to transform
            model_path (Optional[str]): Path to the trained model.
            
        Returns:
            str: The predicted BLTL formula
        """
        # Load the specified model if provided
        if not model_path:
            raise ValueError("Model path is required for prediction")

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.to(device)
        
        # Prepare the input
        if instruction:
            inputs = [instruction + " " + input_text]
        else:
            inputs = [self.prefix + " " + input_text]
            
        inputs = self.tokenizer(
            inputs,
            max_length=self.max_input_length,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        # Generate prediction
        output = self.model.generate(
            **inputs,
            num_beams=8,
            do_sample=True,
            min_length=10,
            max_length=2048
        )
        
        # Decode and print the prediction
        predicted_formula = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        print(predicted_formula.strip())

