"""
author: Difei Tang
email: DIT18@pitt.edu
MeLoDy Lab, University of Pittsburgh

the main script of generating BLTL formulas 
"""

import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Checking.checking import create_property
from bltl_generator import create_bio_bltl_generator
from bltl2nl import generate_bltl_nl_pairs, create_bltl_nl_samples
from bltl_verification import BLTLVerificationPipeline

import logging
from datetime import datetime
from pathlib import Path

output_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "logs"
output_dir.mkdir(parents=True, exist_ok=True)

log_file = output_dir / f'test_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.getLogger('transitions').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w')
logger = logging.getLogger(__name__)

def rephrase_nl(nl: str) -> str:
    """rephrase the NL to be more natural and enrich the dataset"""

    return nl

def test_bltl_generator(num_bltl=None, record_seq=True):
    list_bltl = []

    bltl_generator = create_bio_bltl_generator()
    num_model = len(bltl_generator.biomodels_kb)
    num_round = int(num_bltl / 74) + 1

    seq_nested_bltls = []
    seq_combination_bltls = []
    seq_nested_combination_bltls = []
    for i in range(num_model):
        logger.info('\n')
        logger.info(f'generating BLTLs from the species in the model: {bltl_generator.curr_model["model_name"]}')

        for j in range(num_round):
            block = bltl_generator.generate_bltl()
            bltl = bltl_generator.block_to_string(block)
            logger.info(f'{bltl}')
            
            sequential_vars = bltl_generator.sequential_vars
            number_combination_seq = bltl_generator.number_combination_seq
            number_nested_seq = bltl_generator.number_nested_seq

            if len(sequential_vars) > 0:
                if number_nested_seq == 0 and number_combination_seq == 1:
                    seq_combination_bltls.append(bltl)
                elif number_nested_seq == 1 and number_combination_seq == 0:
                    seq_nested_bltls.append(bltl)
                elif number_nested_seq == 1 and number_combination_seq == 1:
                    seq_nested_combination_bltls.append(bltl)
                else:
                    raise ValueError(f'Invalid number of nested and combination sequences: {number_nested_seq} and {number_combination_seq}')

            list_bltl.append(bltl)

        bltl_generator.update_model_index()

    # for record 
    truncated_bltls = list_bltl[:num_bltl]

    truncated_seq_nested_bltls = [seq_bltl for seq_bltl in seq_nested_bltls if seq_bltl in truncated_bltls]
    truncated_seq_combination_bltls = [seq_bltl for seq_bltl in seq_combination_bltls if seq_bltl in truncated_bltls]
    truncated_seq_nested_combination_bltls = [seq_bltl for seq_bltl in seq_nested_combination_bltls if seq_bltl in truncated_bltls]
    
    truncated_normal_bltls = [bltl for bltl in truncated_bltls if 
                    bltl not in truncated_seq_nested_bltls and 
                    bltl not in truncated_seq_combination_bltls and 
                    bltl not in truncated_seq_nested_combination_bltls]
    
    logger.info('\n')
    logger.info(f'normal: {len(truncated_normal_bltls)}')
    for bltl in truncated_normal_bltls:
        logger.info(f'{bltl}')

    logger.info('\n')
    logger.info(f'seq_nested_combination: {len(truncated_seq_nested_combination_bltls)}')
    for seq_bltl in truncated_seq_nested_combination_bltls:
        logger.info(f'{seq_bltl}')

    logger.info('\n')
    logger.info(f'seq_nested: {len(truncated_seq_nested_bltls)}')

    for seq_bltl in truncated_seq_nested_bltls:
        logger.info(f'{seq_bltl}')

    logger.info('\n')
    logger.info(f'seq_combination: {len(truncated_seq_combination_bltls)}')
    for seq_bltl in truncated_seq_combination_bltls:
        logger.info(f'{seq_bltl}')


    assert len(truncated_bltls) == len(truncated_normal_bltls) + len(truncated_seq_nested_bltls) + \
        len(truncated_seq_combination_bltls) + len(truncated_seq_nested_combination_bltls)

    return truncated_bltls, (truncated_normal_bltls, truncated_seq_nested_bltls, 
            truncated_seq_combination_bltls, truncated_seq_nested_combination_bltls)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate and test BLTL formulas')

    parser.add_argument('--output_name', '-o', type=str, help='Name for the output file')
    parser.add_argument('--num_bltl', '-n', type=int, default=10, help='Number of BLTL formulas to generate')
    args = parser.parse_args()

    args.output_name = 'synbltl_test'

    # generate BLTLs 
    truncated_bltls, detailed_truncated_bltls = test_bltl_generator(num_bltl=args.num_bltl)

    # Testing: verify the BLTLs
    #list_bltl = ['G[23](bdhAB==0&adc==1)&F[23](bdhAB==1)|F[31](spo0A==1&sigK==0)']
    
    bltl_verifier = BLTLVerificationPipeline()
    verified_bltls = bltl_verifier.verify_bltl_batch(truncated_bltls)
    #verified_bltls = list_bltl

    # for record 
    logger.info('\n')
    logger.info(f'verified: {len(verified_bltls)}')
    for verified_bltl in verified_bltls:
        logger.info(f'{verified_bltl}')

    # generate BLTL and NL pairs
    is_diverse = False
    samples = create_bltl_nl_samples()
    #is_diverse = True

    # p1: normal, p2: seq_nested, p3: seq_combination, p4: seq_nested_combination
    sub_ds_names = ['p1', 'p2', 'p3', 'p4'] 
    for bltls, name in zip(detailed_truncated_bltls, sub_ds_names):
        list_nl = generate_bltl_nl_pairs(bltls=bltls, 
                                        output_name=args.output_name + '_' + name,
                                        samples=samples,
                                        use_diverse=is_diverse)

if __name__ == "__main__":
    main()