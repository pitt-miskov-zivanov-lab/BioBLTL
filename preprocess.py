"""
author: Difei Tang
email: DIT18@pitt.edu
MeLoDy Lab, University of Pittsburgh
"""

import os 
import pandas as pd
import json
from pathlib import Path
from typing import List, Optional
import argparse
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import re
import ast 

def remove_whitespace(input_excel_file: str, output_excel_file: str):
    df = pd.read_excel(input_excel_file)
    df['BLTL'] = df['BLTL'].apply(lambda x: re.sub(r'\s+', '', x))
    df.to_excel(output_excel_file, index=False)

def convert_excel_to_jsonl(
    input_excel_file: str,
    output_jsonl: Optional[str] = None,
    id_column: Optional[str] = None,
    enrich_info: bool = False
) -> None:
    """
    Convert a CSV file to JSONL format.
    
    Args:
        input_csv (str): Path to input CSV file
        output_jsonl (str, optional): Path to output JSONL file. If not provided,
                                    will use the same name as input with .jsonl extension
        id_column (str, optional): Column to use as ID. If not provided, will use index
    """
    # Read CSV file
    df = pd.read_excel(input_excel_file)

    if enrich_info:
        df['variables'] = df['variables'].apply(ast.literal_eval)
        df['time_bounds'] = df['time_bounds'].apply(ast.literal_eval)
        columns = ['BLTL', 'new_NL', 'variables', 'time_bounds', 'Type']
    else:
        columns = ['BLTL', 'NL', 'Type']

    if output_jsonl is None:
        output_jsonl = str(Path(input_excel_file).with_suffix('.json'))
    
    # Convert DataFrame to list of dictionaries
    records = df[columns].to_dict('records')
    
    # Write to JSONL file
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for i, record in enumerate(records):
            # Replace NaN values with "None"
            for key, value in record.items():
                if isinstance(value, (list, tuple)):
                    if any(pd.isna(v) for v in value):
                        record[key] = "None"
                else:
                    if pd.isna(value):
                        record[key] = "None"

            # Add ID if specified
            if id_column and id_column in record:
                record['id'] = str(record[id_column])
            else:
                record['id'] = str(i)
                
            # Write each record as a JSON line
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(records)} records to {output_jsonl}")

def concat_jsonl_files(
    input_files: List[str],
    output_file: str,
    reset_ids: bool = True
) -> None:
    """
    Concatenate multiple JSONL files into a single JSONL file.
    
    Args:
        input_files (List[str]): List of input JSONL file paths
        output_file (str): Path to output concatenated JSONL file
        reset_ids (bool): Whether to reset IDs in sequential order
    """
    all_records = []
    
    # Read all input files
    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())
                all_records.append(record)
    
    # Write concatenated records to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, record in enumerate(all_records):
            if reset_ids:
                record['id'] = i
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"Concatenated {len(all_records)} records from {len(input_files)} files to {output_file}")

def concat_json_files(input_files: List[str], output_file: str):
    """
    Concatenate multiple JSON files into a single JSON file.
    """
    all_records = []    
    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())
                all_records.append(record)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Concatenated {len(all_records)} records from {len(input_files)} files to {output_file}")

def convert_jsonl_to_alpaca(input_jsonl_file: str, output_alpaca_file: str, version: str = None):
    """
    Convert JSONL file to Alpaca format.
    """
    alpaca_records = []
    with open(input_jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())

            # Convert to Alpaca format
            # FIXME: history is not used
            if 'new_NL' in record:
                nl = record['new_NL']
            else:
                nl = record['NL']

            suffix = "Transform the following sentence into a Bounded Linear Temporal Logic (BLTL) formula."

            if version:            
                variables = ', '.join(record['variables'])
                time_bounds = ', '.join(map(str, record['time_bounds']))
                # suffix += f' Variables: {variables}.'
                # suffix += f' Time bounds: {time_bounds}.'

                alpaca_record = {
                    "instruction": f"{suffix}",
                    "input": nl,
                    "output": record['BLTL'],
                    "type": record['Type'],
                    "variables": record['variables'],
                    "time_bounds": record['time_bounds'],
                }
            else:
                alpaca_record = {
                    "instruction": f"{suffix}",
                    "input": nl,
                    "output": record['BLTL'],
                    "type": record['Type']
                }

            alpaca_records.append(alpaca_record)
    
    with open(output_alpaca_file, 'w', encoding='utf-8') as f:
        for record in alpaca_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

def split_data(input_jsonl_file: str, output_dir: str, train_ratio: float = 0.6, seed: int = 42):
    with open(input_jsonl_file, 'r') as f:
        data = [json.loads(line) for line in f]

    # train, dev, test = 0.6, 0.2, 0.2
    train_data, temp_data = train_test_split(data, train_size=train_ratio, random_state=seed)
    dev_data, test_data = train_test_split(temp_data, train_size=0.5, random_state=seed)

    # Save
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.basename(input_jsonl_file)
    filename = os.path.splitext(filename)[0]
    def save_jsonl(filename, dataset):
        with open(os.path.join(output_dir, filename), 'w') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    save_jsonl(f'{filename}_train.json', train_data)
    save_jsonl(f'{filename}_eval.json', dev_data)
    save_jsonl(f'{filename}_test.json', test_data)

    print(f" Split complete:")
    print(f" Train: {len(train_data)}")
    print(f" Dev:   {len(dev_data)}")
    print(f" Test:  {len(test_data)}")

    return (f'{filename}_train.json', f'{filename}_eval.json', f'{filename}_test.json')

def split_biobltl_data(input_jsonl_file: str, output_dir: str, train_ratio: float = 0.7, seed: int = 42):
    """
    Split biobltl data into train and test sets 
    """
    with open(input_jsonl_file, 'r') as f:
        data = [json.loads(line) for line in f]

    # train, dev, test = 0.6, 0.2, 0.2
    train_data, test_data = train_test_split(data, train_size=train_ratio, random_state=seed)

    # Save
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.basename(input_jsonl_file)
    filename = os.path.splitext(filename)[0]
    def save_jsonl(filename, dataset):
        with open(os.path.join(output_dir, filename), 'w') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    save_jsonl(f'{filename}_train.json', train_data)
    save_jsonl(f'{filename}_test.json', test_data)

    print(f" Split complete:")
    print(f" Train: {len(train_data)}")
    print(f" Test:  {len(test_data)}")

# def split_gold_data(input_jsonl_file: str, output_dir: str, train_ratio: float = 0.7, seed: int = 42):
#     """
#     Split biobltl data into train and test sets 
#     """
#     with open(input_jsonl_file, 'r') as f:
#         data = [json.loads(line) for line in f]

#     train_data, temp_data = train_test_split(data, train_size=train_ratio, random_state=seed)
#     _, test_data = train_test_split(temp_data, train_size=0.5, random_state=seed)
    
#     # Create KFold cross validator
#     kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    
#     # Save
#     os.makedirs(output_dir, exist_ok=True)
#     filename = os.path.basename(input_jsonl_file)
#     filename = os.path.splitext(filename)[0]
    
#     def save_jsonl(filename, dataset):
#         with open(os.path.join(output_dir, filename), 'w') as f:
#             for item in dataset:
#                 f.write(json.dumps(item, ensure_ascii=False) + '\n')

#     # Save test set
#     save_jsonl(f'{filename}_test.json', test_data)

#     print(f"\nSplit complete:")
#     print(f"Total samples: {len(data)}")
#     print(f"Test set: {len(test_data)} samples")
    
#     # Generate and save cross validation folds
#     for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
#         fold_train_data = [train_data[i] for i in train_idx]
#         fold_val_data = [train_data[i] for i in val_idx]
        
#         # Save fold data
#         save_jsonl(f'{filename}_train{fold}.json', fold_train_data)
#         save_jsonl(f'{filename}_eval{fold}.json', fold_val_data)

#         print(f"Fold {fold+1}:")
#         print(f"  Train: {len(fold_train_data)} samples")
#         print(f"  Val:   {len(fold_val_data)} samples")

def preprocess_biobltl_data():
    # prepare the gold standard data
    _dir = 'preprocessed-vars-bounds-gcd'
    input_excel_files = [f'data/gold/{_dir}/Tcell_Natasa_curated_preprocessed.xlsx', f'data/gold/{_dir}/PCC_Qinsi_curated_preprocessed.xlsx']

    for f in input_excel_files:
        convert_excel_to_jsonl(input_excel_file=f, enrich_info=True)

    convert_jsonl_to_alpaca(input_jsonl_file=f'data/gold/{_dir}/Tcell_Natasa_curated_preprocessed.json',
                             output_alpaca_file=f'data/gold/{_dir}/tcell_bltl_v2.json', version='v2')
    convert_jsonl_to_alpaca(input_jsonl_file=f'data/gold/{_dir}/PCC_Qinsi_curated_preprocessed.json', 
                            output_alpaca_file=f'data/gold/{_dir}/pcc_bltl_v2.json', version='v2')

    #split_biobltl_data(input_jsonl_file='data/gold/tcell_bltl.json', output_dir='data/gold')

def preprocess_synbltl_data():
    convert_jsonl_to_alpaca(input_jsonl_file='./data/generated/synbltl_gen_1000_4o.json', output_alpaca_file='data/generated/synbltl_1000_4o.json')

    split_data(input_jsonl_file='data/generated/synbltl_1000_4o.json', output_dir='data/generated')

def preprocess_synbltl_data_v2(input_basename: str, input_dir: str, output_dir: str):

    input_file_name = os.path.join(input_dir, input_basename)
    input_file_name2 = os.path.splitext(input_file_name)[0] # remove .json

    sub_ds_names = ['p1', 'p2', 'p3', 'p4']

    sub_ds_split_train_files = []
    sub_ds_split_eval_files = []
    sub_ds_split_test_files = []

    all_alpaca_files = []
    for name in sub_ds_names:
        sub_ds_name = input_file_name2 + '_' + name
        alpaca_file_name = sub_ds_name + '_alpaca.json'

        convert_jsonl_to_alpaca(input_jsonl_file=sub_ds_name + '.json', 
                                type=name,
                                output_alpaca_file=alpaca_file_name)
        all_alpaca_files.append(alpaca_file_name)

        train_file, eval_file, test_file = split_data(input_jsonl_file=alpaca_file_name, output_dir=output_dir)

        # hard-code here 
        sub_ds_split_train_files.append(os.path.join(output_dir, train_file))
        sub_ds_split_eval_files.append(os.path.join(output_dir, eval_file))
        sub_ds_split_test_files.append(os.path.join(output_dir, test_file))
    
    # subsets
    concat_jsonl_files(input_files=sub_ds_split_train_files, output_file=output_dir + '/' + input_basename.replace('.json', '') + '_train.json')
    concat_jsonl_files(input_files=sub_ds_split_eval_files, output_file=output_dir + '/' + input_basename.replace('.json', '') + '_eval.json')
    concat_jsonl_files(input_files=sub_ds_split_test_files, output_file=output_dir + '/' + input_basename.replace('.json', '') + '_test.json')

    # full 
    concat_jsonl_files(input_files=all_alpaca_files, output_file=output_dir + '/' + input_basename)

def concat_pcc_tcell_data(): 
    folder_names = {'ori': 'ori', 'preprocessed': 'v0',
                    'preprocessed-vars-bounds': 'v1',
                    'preprocessed-vars-bounds-gcd': 'v2'}
    
    for folder_name, version in folder_names.items():
        _dir = './data/gold/' + folder_name
        input_pcc_file = f'pcc_bltl_{version}.json'
        input_tcell_file = f'tcell_bltl_{version}.json'
        input_files = [os.path.join(_dir, input_pcc_file), os.path.join(_dir, input_tcell_file)]    
        output_file = f'pcc_tcell_bltl_{version}.json'
        concat_json_files(input_files=input_files, output_file=os.path.join(_dir, output_file))

def main():
    #preprocess_synbltl_data()
    #preprocess_biobltl_data()

    #input_dir = './data/generated/ori'
    # input_basename = 'synbltl_1000_s5.json'
    # output_dir = './data/generated/release/' + input_basename.replace('.json', '')

    # preprocess_synbltl_data_v2(input_basename=input_basename, input_dir=input_dir, output_dir=output_dir)

    concat_pcc_tcell_data()
    pass 

if __name__ == "__main__":
    main()