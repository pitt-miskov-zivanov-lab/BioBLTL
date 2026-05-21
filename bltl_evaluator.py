"""
author: Difei Tang
email: DIT18@pitt.edu
MeLoDy Lab, University of Pittsburgh
"""

import os
import sys
from pathlib import Path
import subprocess
from enum import IntEnum
import json

class ExitCode(IntEnum):
    EXIT_FAILURE = 1
    EXIT_SUCCESS = 0

    EXIT_SUCCESS_EQUIVALENT = 48
    EXIT_SUCCESS_DIFFERENT = 49
    EXIT_SUCCESS_VALID = 50
    EXIT_SUCCESS_SYNTAX_ERROR = 51

class BLTLEvaluator:
    def __init__(self):
        self.checking_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Checking")
        self.checking_dir = Path(self.checking_dir)

        self.parser_path = self.checking_dir / "dishwrap_v1.0/monitor/bltl_parser"
        self.evaluator_path = self.checking_dir / "dishwrap_v1.0/monitor/bltl_evaluator"
        
        # make sure the executables exist
        if not self.parser_path.exists():
            raise FileNotFoundError(f"Parser executable not found at {self.parser_path}")
        if not self.evaluator_path.exists():
            raise FileNotFoundError(f"Evaluator executable not found at {self.evaluator_path}")

    def evaluate_bltl(self, bltl_1: str, bltl_2: str):
        """evaluate the BLTL formula"""
        try:
            # create a temporary file to store the BLTL formula
            temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
            os.makedirs(temp_dir, exist_ok=True)

            property_file_1 = os.path.join(temp_dir, 'temp1.bltl')
            with open(property_file_1, 'w') as f:
                f.write(bltl_1)

            property_file_2 = os.path.join(temp_dir, 'temp2.bltl')
            with open(property_file_2, 'w') as f:
                f.write(bltl_2)

            # run the parser
            syntax_result_1 = self._run_parser(property_file_1)
            syntax_result_2 = self._run_parser(property_file_2)

            # run the evaluator
            evaluator_result = self._run_evaluator(property_file_1, property_file_2)

            if not syntax_result_1['valid']:
                return False
            if not syntax_result_2['valid']:
                return False

            return evaluator_result

        except Exception as e:
            # we assume the parser and evaluator can be run successfully
            raise ValueError(f"Evaluation failed: {str(e)}")
        
        finally:
            # clean up the temporary files
            Path(property_file_1).unlink(missing_ok=True)
            Path(property_file_2).unlink(missing_ok=True)

    def _run_parser(self, property_file: str):
        try:
            result = subprocess.run(
                [str(self.parser_path), property_file],
                capture_output=True,
                text=True,
                check=False
            )
            
            # parse the output
            if result.returncode == ExitCode.EXIT_SUCCESS_VALID:
                return {'valid': True}
            elif result.returncode == ExitCode.EXIT_SUCCESS_SYNTAX_ERROR:
                return {'valid': False, 'error': "Syntax error in BLTL formula"}
            else:
                return {'valid': False, 'error': result.stderr.strip() or "Unknown parser error"}
                
        except subprocess.SubprocessError as e:
            return {'valid': None, 'error': f"Parser execution failed: {str(e)}"}

    def _run_evaluator(self, property_file: str, trace_file: str):
        try:
            result = subprocess.run(
                [str(self.evaluator_path), property_file, trace_file],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == ExitCode.EXIT_SUCCESS_EQUIVALENT:
                return {'equivalent': True}
            elif result.returncode == ExitCode.EXIT_SUCCESS_DIFFERENT:
                return {'equivalent': False}
            else:
                return {
                    'equivalent': None,
                    'error': result.stderr.strip() or "Unknown evaluator error"
                }
        except subprocess.SubprocessError as e:
            return {'equivalent': None, 'error': f"Evaluator execution failed: {str(e)}"}
        
def test_evaluator():
    evaluator = BLTLEvaluator()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, './data/generated_data_Apr14.jsonl'), 'r') as f:
        for i, line in enumerate(f):
            if i > 100:
                break
            data = json.loads(line)
            bltl_1 = data['output']
            bltl_2 = data['output']
            evaluated_result = evaluator.evaluate_bltl(bltl_1, bltl_2)

            if not evaluated_result:
                print(f"error: BLTL 1: {bltl_1} and BLTL 2: {bltl_2} parsing failed")
                continue
            
            if not evaluated_result['equivalent']:
                print(f"warning: BLTL 1: {bltl_1} and BLTL 2: {bltl_2} are not equivalent")

            if evaluated_result['equivalent']:
                print(f"success!")

def main():
    pass 
    #test_evaluator()

if __name__ == "__main__":
    main()