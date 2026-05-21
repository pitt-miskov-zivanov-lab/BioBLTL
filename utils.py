import json
from typing import List
import os
import math 
import random

from sympy import N

# manually fix the distribution of the data 
NUM_FIX_RECORDS = 8
FIX_RECORDS = []

def concat_json_files(input_files: List[str], output_file: str, subset_name: str):
    """
    Concatenate multiple JSON files into a single JSON file.
    """
    all_records = []    
    is_fixed = False
    
    current_id = 0

    global NUM_FIX_RECORDS
    global FIX_RECORDS

    def add_record(record: dict, current_id: int) -> None:
        record['id'] = current_id
        all_records.append(record.copy())
        return current_id + 1

    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())

                if subset_name == 'test' and record['type'] == 'p1' and len(FIX_RECORDS) < NUM_FIX_RECORDS:
                    FIX_RECORDS.append(record.copy())
                    continue
                elif subset_name == 'eval' and not is_fixed:
                    # add 1 p1 record to eval
                    current_id = add_record(FIX_RECORDS.pop(), current_id)
                    is_fixed = True
                elif subset_name == 'train' and not is_fixed:
                    # add 7 p1 records to train 
                    for i in range(len(FIX_RECORDS)):
                        current_id = add_record(FIX_RECORDS.pop(), current_id)

                    is_fixed = True
                
                current_id = add_record(record, current_id)

    # sanity check
    ids = set([record['id'] for record in all_records])
    assert len(ids) == len(all_records)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

def split_json_file(input_file: str, output_file: str, number_examples: int=100):
    """
    Split a JSON file into multiple files.
    """

    random.seed(42)

    records_by_split = {'p1': [], 'p2': [], 'p3': [], 'p4': []}
    type_count = {'p1': 0, 'p2': 0, 'p3': 0, 'p4': 0}

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            records_by_split[record['type']].append(record)
            type_count[record['type']] += 1

    ratios = [count / sum(type_count.values()) for count in type_count.values()]
    target_ratios = [math.ceil(ratio * number_examples) for ratio in ratios]

    assert sum(target_ratios) >= number_examples

    targets = {split: target for split, target in zip(type_count.keys(), target_ratios)}

    # Initialize splits
    splits = {'p1': [], 'p2': [], 'p3': [], 'p4': []}
    total_count = 0

    for i in range(number_examples):
        if total_count >= number_examples:
            break

        for split, count in targets.items(): 

            if not records_by_split[split]:
                raise ValueError(f"No records left for split: {split}")
            
            if count == 0:
                continue

            chosen_record = random.choice(records_by_split[split])
            records_by_split[split].remove(chosen_record)
            splits[split].append(chosen_record)
            targets[split] -= 1
            total_count += 1

    with open(output_file, 'w', encoding='utf-8') as f:
        for _, records in splits.items():
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

def split_train_data(): 
    main_dir = './data/generated/release'
    input_basename = 'synbltl_5000'
    input_dir = os.path.join(main_dir, input_basename)
    
    output_dir = input_dir 
    os.makedirs(output_dir, exist_ok=True)
    
    for subset_number in [100, 500, 1000, 2000]:

        input_file = os.path.join(input_dir, input_basename + '_train.json')
        output_file = os.path.join(output_dir, f'synbltl_{subset_number}_train.json')
        
        split_json_file(input_file=input_file, output_file=output_file, number_examples=subset_number)

# def split_data_v0(): 
#     main_dir = './data/generated/release'
#     input_basename = 'synbltl_5000_release'
#     input_dir = os.path.join(main_dir, input_basename)
#     number_examples = 100

#     output_dir = os.path.join(main_dir, f'synbltl_{number_examples}_release')
#     os.makedirs(output_dir, exist_ok=True)

#     subset_split_ratios = {'train': 0.6, 'eval': 0.2, 'test': 0.2}
#     for subset_name, ratio in subset_split_ratios.items():
#         subset_number = int(ratio * number_examples)

#         input_file = os.path.join(input_dir, input_basename + '_' + subset_name + '.json')
#         output_file = os.path.join(output_dir, f'synbltl_{number_examples}_s1_{subset_name}.json')
        
#         split_json_file(input_file=input_file, output_file=output_file, number_examples=subset_number)

# def concat_data_v0():
#     main_dir = './data/generated/release'

#     def get_file_name(input_basename: str):
#         return os.path.join(main_dir, input_basename, input_basename)

#     input_basename1 = 'synbltl_1000_s1'
#     input_file1 = get_file_name(input_basename1)

#     input_basename2 = 'synbltl_1000_s2'
#     input_file2 = get_file_name(input_basename2)

#     output_basename = 'synbltl_2000'
#     output_dir = os.path.join(main_dir, output_basename)
#     os.makedirs(output_dir, exist_ok=True)

#     output_file = get_file_name(output_basename)
    
#     subset_split_names = ['train', 'eval', 'test']
#     for subset_name in subset_split_names:
#         file1 = input_file1 + f'_{subset_name}.json'
#         file2 = input_file2 + f'_{subset_name}.json'

#         concat_json_files(input_files=[file1, file2], 
#                           output_file=f'{output_file}_{subset_name}.json')

def concat_data_v1():
    """we will use this function to get the full data and then split into subsets"""
    main_dir = './data/generated/release'

    def get_file_name(input_basename: str):
        return os.path.join(main_dir, input_basename, input_basename)


    output_basename = 'synbltl_5000_release'
    output_dir = os.path.join(main_dir, output_basename)

    os.makedirs(output_dir, exist_ok=True)

    subset_split_names = ['test', 'eval', 'train']
    for subset_name in subset_split_names:

        input_files = []
        for i in ['s1', 's2', 's3', 's4', 's5']:
            input_basename = f'synbltl_1000_{i}'
            input_file = get_file_name(input_basename) + f'_{subset_name}.json'
            input_files.append(input_file)

        concat_json_files(input_files=input_files, 
                          output_file=f'{output_dir}/synbltl_5000_{subset_name}.json', 
                          subset_name=subset_name)

def main():
    split_train_data()
    #concat_data_v1()

if __name__ == '__main__':
    main()
