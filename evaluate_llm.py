"""
author: Difei Tang
email: DIT18@pitt.edu
MeLoDy Lab, University of Pittsburgh

Report LLM inference results
"""

import argparse
from pathlib import Path
from bltl_evaluator import BLTLEvaluator
import json
from typing import List, Tuple
import nltk
import os
import numpy as np
from tqdm import tqdm
import re
import math

from evaluate import load as load_metric

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

def extract_nl_from_prompt(prompt_text, type: str = 'None'):
    # Pattern to match the NL sentence between the instruction and the end tag
    if type == 'qwen':
        pattern = r'Transform the following sentence into a Bounded Linear Temporal Logic \(BLTL\) formula\.\n(.*?)<\|im_end\|>'
    elif type == 'instruction':
        pattern = r'Transform the following sentence into (?:a )?Bounded Linear Temporal Logic \(BLTL\) formula\.?\s*:?\s*\n?(.*?)(?:INPUT:|OUTPUT:|$)'
    else: 
        return prompt_text

    match = re.search(pattern, prompt_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def validate_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            try:
                json.loads(line.strip())
            except json.JSONDecodeError:
                print(f"Error in line {i}: {line}")
                return False
    return True

# def bootstrap_bleu_hf(preds: List[str], refs: List[str], n_bootstrap: int = 1000) -> Tuple[float, float, float]:
#     assert len(preds) == len(refs)
#     n = len(preds)
#     bleu_scores = []

#     for _ in tqdm(range(n_bootstrap), desc="Bootstrap sampling for BLEU"):
#         indices = np.random.choice(n, size=n, replace=True)
#         preds_b = [preds[i] for i in indices]
#         refs_b = [refs[i] for i in indices]

#         # HuggingFace evaluate BLEU expects refs as [[ref1], [ref2], ...]
#         score_dic = bleu_metric.compute(
#             predictions=preds_b,
#             references=[[r] for r in refs_b],
#             smooth_method="exp",
#             tokenize="13a"
#         )
#         bleu_scores.append(score_dict["score"] / 100.0)

#     mean_bleu = np.mean(bleu_scores)
#     ci_low, ci_high = np.percentile(bleu_scores, [2.5, 97.5])
#     return mean_bleu, ci_low, ci_high

def wilson_accuracy_ci(results: List[bool], alpha: float = 0.05) -> Tuple[float, float, float]:
    """
    Args:
        results: Boolean list, True indicates correct prediction
        alpha: Significance level (0.05 -> 95% CI)
        
    Returns:
        (acc, ci_low, ci_high) 
    """
    n = len(results)
    if n == 0:
        return 0.0, 0.0, 0.0

    k = int(np.sum(results))
    p = k / n

    z = 1.959963984540054 if alpha == 0.05 else abs(np.quantile(np.random.standard_normal(1_000_000), 1 - alpha/2))
    
    denom = 1 + (z**2)/n
    center = (p + (z**2)/(2*n)) / denom
    half = (z / denom) * math.sqrt((p*(1 - p))/n + (z**2)/(4*(n**2)))
    
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return p, lo, hi

def bootstrap_accuracy(results: List[bool], n_bootstrap=1000) -> tuple:
    """
    Args:
        results: List of boolean values indicating correct/incorrect predictions
        n_bootstrap: Number of bootstrap samples
    
    Returns:
        tuple: (mean accuracy, standard deviation)
    """
    accuracies = []
    n = len(results)
    
    for _ in tqdm(range(n_bootstrap), desc="Bootstrap sampling"):
        # Random sampling with replacement
        indices = np.random.choice(n, size=n, replace=True)
        bootstrap_results = [results[i] for i in indices]
        accuracy = sum(bootstrap_results) / len(bootstrap_results)
        accuracies.append(accuracy)
    
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    return mean_acc, std_acc

def normalize_formula(formula: str) -> str:
    formula = re.sub(r'F\[(\d+)\]', r'F[\1]', formula)
    formula = re.sub(r'G\[(\d+)\]', r'G[\1]', formula)
    formula = re.sub(r'U\[(\d+)\]', r'U[\1]', formula)
    
    def lower_var(match):
        return match.group(0).lower()
    
    formula = re.sub(r'[A-Za-z_][A-Za-z0-9_]*(?= *[=<>])', lower_var, formula)
    
    return formula

def extract_all_nested_parentheses(formula):
    """Extract ALL parentheses content including deeply nested ones."""
    
    def find_all_parentheses_recursive(text, level=0):
        """Recursively find all parentheses at all levels."""
        results = []
        
        # Pattern to match any parentheses content (including nested)
        pattern = r'\(([^()]*(?:\([^()]*\)[^()]*)*)\)'
        
        matches = list(re.finditer(pattern, text))
        for match in matches:
            content = match.group(1)
            results.append((content, level))
            
            # Recursively find nested content
            nested_results = find_all_parentheses_recursive(content, level + 1)
            results.extend(nested_results)
        
        return results
    
    all_matches = find_all_parentheses_recursive(formula)

    filtered_matches = []
    for content, level in all_matches:
        if 'F[' not in content and 'G[' not in content:
            filtered_matches.append(content)

    return filtered_matches

def sort_atomic_propositions(content):
    """Sort atomic propositions within a clause."""
    content = content.strip()
    
    if '&' in content and '|' in content:
        # Mixed operators - manully check
        print(f'Mixed operators in: {content}')
        return content  
    elif '&' in content:
        aps = [ap.strip() for ap in content.split('&')]
        sorted_aps = sorted(aps, key=lambda x: x.split('==')[0].strip())
        return '&'.join(sorted_aps)
    elif '|' in content:
        aps = [ap.strip() for ap in content.split('|')]
        sorted_aps = sorted(aps, key=lambda x: x.split('==')[0].strip())
        return '|'.join(sorted_aps)
    else:
        # Single atomic proposition
        return content

def reorder_nested_aps(formula):
    """reorder nested atomic propositions to avoid unmatching issues"""
    contents = extract_all_nested_parentheses(formula)
    
    for content in contents:
        sorted_content = sort_atomic_propositions(content)

        old_pattern = f"({content})"
        new_pattern = f"({sorted_content})"
        if not new_pattern == old_pattern:
            formula = formula.replace(old_pattern, new_pattern)

    return formula

def evaluate_results(output_list: List[dict], output_dir: Path, bootstrap_sampling: bool = False) -> None:
    evaluator = BLTLEvaluator()

    correct_results = []
    wrong_results = []
    all_results = []
    count_correct = 0
    count_parser_valid = 0

    preds_for_bleu = []
    refs_for_bleu = []

    bleu_metric = load_metric("sacrebleu")

    for i, output in enumerate(output_list):
        prompt = output['prompt']
        predicted_str = output['predict']
        label_str = output['label']

        # retrieve bleu score like llamafactory will do 
        preds_for_bleu.append(predicted_str.strip())
        refs_for_bleu.append(label_str.strip())
       
        pred = normalize_formula(predicted_str.strip())
        label = normalize_formula(label_str.strip())

        is_correct = False
        match_type = "none"
        parser_valid = False

        pred = reorder_nested_aps(pred)
        label = reorder_nested_aps(label)

        if 'manual' in output.keys():
            count_correct += 1
            is_correct = True
            match_type = "manual"
            parser_valid = True
            count_parser_valid += 1
        elif pred == label:
            # FIXME: not sure if this is correct for evaluation
            count_correct += 1
            is_correct = True
            match_type = "exact"
            parser_valid = True
            count_parser_valid += 1
        else:
            evaluated_result = evaluator.evaluate_bltl(pred, label)
            if evaluated_result:
                parser_valid = True
                count_parser_valid += 1
                if evaluated_result['equivalent']:
                    count_correct += 1
                    is_correct = True
                    match_type = "equivalent"
            else:
                parser_valid = False

        # FIXME: qwen prompt template 
        nl_str = extract_nl_from_prompt(prompt)
        print('prompt: ', nl_str)
        print('pred:', predicted_str)
        print('label:', label_str)

        score_dict = bleu_metric.compute(
            predictions=preds_for_bleu,
            references=[[r] for r in refs_for_bleu],   # list of list
            smooth_method="exp",
            tokenize="13a"                             # same defaults as LLaMA-Factory
        )
        bleu_score = score_dict["score"] / 100.0 

        result = {
            'NL': nl_str,
            'predicted': pred,
            'actual': label,
            'match_type': match_type,
            'parser_valid': parser_valid,
        }

        all_results.append(is_correct)
        if is_correct:
            correct_results.append(result)
        else:
            wrong_results.append(result)


    if bootstrap_sampling:
        # calculate accuracy and confidence interval
        mean_acc, std_acc = bootstrap_accuracy(all_results)
        accuracy = f"{mean_acc*100:.2f} ± {std_acc*100:.2f}%"
        print(f'Bootstrap Accuracy: {accuracy}\n')

        acc, lo, hi = wilson_accuracy_ci(all_results)
        half = (hi - lo) / 2 
        accuracy = f'{acc*100:.2f} ± {half*100:.2f}%'
        print(f'Wilson Accuracy: {accuracy}\n')
    else:
        accuracy = count_correct / len(output_list)
        parser_valid_accuracy = count_parser_valid / len(output_list)

        print(f'BLEU Score: {bleu_score*100:.2f}%\n')
        print(f'Accuracy: {accuracy*100:.2f}%\n')
        print(f'Parser Valid Accuracy: {parser_valid_accuracy*100:.2f}%\n')
    
    # Save results to file
    with open(output_dir / f'accuracy.txt', 'w') as f:
        f.write(f'accuracy is: {accuracy*100:.2f}%\n') 
        f.write(f'parser valid accuracy is: {parser_valid_accuracy*100:.2f}%\n')
        f.write(f'bleu score is: {bleu_score*100:.2f}%\n')

    with open(output_dir / f'correct_results.txt', 'w') as fc:
        for result in correct_results:
            fc.write(f"NL: {result['NL']}\n")
            fc.write(f"Predicted: {result['predicted']}\n")
            fc.write(f"Actual: {result['actual']}\n")
            fc.write(f"Match Type: {result['match_type']}\n")
            fc.write(f"Parser Valid: {result['parser_valid']}\n")
            fc.write("\n")

    with open(output_dir / f'wrong_results.txt', 'w') as fw:
        for result in wrong_results:
            fw.write(f"NL: {result['NL']}\n")
            fw.write(f"Predicted: {result['predicted']}\n")
            fw.write(f"Actual: {result['actual']}\n")
            fw.write(f"Match Type: {result['match_type']}\n")
            fw.write(f"Parser Valid: {result['parser_valid']}\n")
            fw.write("\n")

    return accuracy

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--bootstrap_sampling', type=bool, default=False,
                      help='Use bootstrap sampling for accuracy calculation')
    parser.add_argument('--output_dir', type=str, default=None,
                      help='Path to the results directory')
    parser.add_argument('--multi_run_dir', type=str, default=None,
                        help='Path to directory containing multiple inference runs')
    
    args = parser.parse_args()

    if args.multi_run_dir:
        all_accuracy = []

        multi_run_dir = Path(args.multi_run_dir)
        for run_dir in multi_run_dir.glob('run_*'):
            run_id = run_dir.name.split('_')[-1]
            results_path = run_dir / 'generated_predictions.jsonl'
            if not validate_jsonl(results_path):
                print(f"Invalid JSONL file in {run_dir}. Skipping.")
                continue

            print(f"\nEvaluating run_{run_id}")
            with open(results_path, 'r') as f:
                output_list = [json.loads(line) for line in f]

            accuracy = evaluate_results(output_list, run_dir)
            all_accuracy.append(accuracy)

        mean_acc = np.mean(all_accuracy)
        std_acc = np.std(all_accuracy)
        accuracy = f"{mean_acc*100:.2f} ± {std_acc*100:.2f}%"
        print(f"Final Accuracy with multiple runs: {accuracy}")

    if args.output_dir:
        output_dir = Path(args.output_dir)

        results_path = output_dir / 'generated_predictions.jsonl'
        if not validate_jsonl(results_path):
            print("Invalid JSONL file. Exiting.")
            return
    
        with open(results_path, 'r') as f:
            output_list = [json.loads(line) for line in f]

        if args.bootstrap_sampling:
            evaluate_results(output_list, output_dir, bootstrap_sampling=True)
        else:
            evaluate_results(output_list, output_dir)

if __name__ == "__main__":
    main()