from pathlib import Path
import subprocess
import tempfile
from typing import Dict, Optional
from dataclasses import dataclass
from enum import IntEnum
import os 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pydantic
from dotenv import load_dotenv

load_dotenv()

class ExitCode(IntEnum):
    EXIT_FAILURE = 1
    EXIT_SUCCESS = 0

    EXIT_SUCCESS_VALID = 50
    EXIT_SUCCESS_SYNTAX_ERROR = 51
    EXIT_SUCCESS_SAT = 52
    EXIT_SUCCESS_UNSAT = 53

@dataclass
class BLTLVerificationResult:
    is_valid: bool
    syntax_ok: bool
    satisfied: Optional[bool] = None
    error_message: Optional[str] = None
    suggestion: Optional[str] = None

class RefinedBLTL(pydantic.BaseModel):
    refined_bltl: str

class BLTLVerificationPipeline:
    def __init__(self):

        self.checking_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Checking")
        self.checking_dir = Path(self.checking_dir)

        self.parser_path = self.checking_dir / "dishwrap_v1.0/monitor/bltl_parser"
        self.checker_path = self.checking_dir / "dishwrap_v1.0/monitor/bltl_checker"
        
        # make sure the executables exist
        if not self.parser_path.exists():
            raise FileNotFoundError(f"Parser executable not found at {self.parser_path}")
        if not self.checker_path.exists():
            raise FileNotFoundError(f"Checker executable not found at {self.checker_path}")
        
        # load the LLM as we need it for refinement
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("LLM_MODEL")
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model=self.model,
            temperature=0
        )

        self.init_verification_chains()
        self.init_post_refinement_chains()

    def init_verification_chains(self):
        """initialize the verification chains"""

        system_prompt = """
        You are a helpful assistant that refines BLTL formulas based on the issues found.

        The formula contains temporal operators with the following meanings:
        - F[t] φ: Finally: φ eventually has to hold (some where on the subsequent path).
        - G[t] φ: Globally: φ has to hold on the entire subsequent path.
        - ρ U[t] φ: Until: ρ has to hold at least until φ, which holds at the current or a future state.

        """
        user_prompt = """
        Original BLTL: {bltl}
        Issues found: {issues}
        Please provide the refined BLTL.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt)
        ])

        structured_model = self.llm.with_structured_output(RefinedBLTL)
        self.verification_chain = prompt | structured_model

    def init_post_refinement_chains(self):
        system_prompt = """You are an expert at refining BLTL formulas.
        Given a natural language description and the predicted BLTL formula, fix the common errors in the predicted formula.    
        Focus on fixing time bounds, variable names, and logical structure.
        Refine the BLTL to be more accurate and correct and match the natural language description.
        """

        user_prompt = """
        NL: {nl}
        BLTL: {bltl}

        Please provide the refined version of the predicted formula:
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt)
        ])

        structured_model = self.llm.with_structured_output(RefinedBLTL)
        self.post_refinement_chain = prompt | structured_model
        
    def verify_bltl(self, bltl: str, trace_file: str = None) -> BLTLVerificationResult:
        """verify the complete pipeline of BLTL formula"""
        try:
            # create a temporary file to store the BLTL formula
            temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
            os.makedirs(temp_dir, exist_ok=True)

            property_file = os.path.join(temp_dir, 'temp.bltl')
            with open(property_file, 'w') as f:
                f.write(bltl)

            # run the parser
            syntax_result = self._run_parser(property_file)
            return syntax_result

        except Exception as e:
            # we assume the parser can be run successfully
            raise ValueError(f"Verification failed: {str(e)}")
        
        finally:
            # clean up the temporary file 
            Path(property_file).unlink(missing_ok=True)

    def _run_parser(self, property_file: str) -> Dict:
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

    def _run_checker(self, property_file: str, trace_file: str) -> Dict:
        try:
            result = subprocess.run(
                [str(self.checker_path), property_file, trace_file],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == ExitCode.EXIT_SUCCESS_SAT:
                return {'satisfied': True}
            elif result.returncode == ExitCode.EXIT_SUCCESS_UNSAT:
                return {'satisfied': False}
            else:
                return {
                    'satisfied': None,
                    'error': result.stderr.strip() or "Unknown checker error"
                }
        except subprocess.SubprocessError as e:
            return {'satisfied': None, 'error': f"Checker execution failed: {str(e)}"}
        
    def refine_bltl(self, bltl, issues):
        """refine the BLTL formula based on the issues"""
        result = self.verification_chain.invoke({"bltl": bltl, "issues": issues})

        return result.refined_bltl
    
    def refine_bltl_posthoc(self, nl, bltl):
        """refine the BLTL formula based on the natural language description"""

        attempts = 0
        while attempts < 2:
            result = self.post_refinement_chain.invoke({
                    "nl": nl, 
                    "bltl": bltl
                })
            
            bltl = result.refined_bltl
            verification_result = self.verify_bltl(bltl)
            if verification_result['valid']:
                return bltl
            else: 
                print(f"Refining BLTL to fix the syntax error, attempt {attempts + 1}")
                bltl = self.refine_bltl(bltl, verification_result['error'])
            
            attempts += 1

        return bltl
    
    def verify_bltl_batch(self, bltls):
        """verify a batch of BLTL formulas"""
        verified_bltls = []
        for bltl in bltls:
            verification_result = self.verify_bltl(bltl)
            if verification_result['valid']:
                verified_bltls.append(bltl) 
            else:
                raise ValueError(f"BLTL formula is invalid: {bltl}")
        
        return verified_bltls

    def process_bltl_batch(self, bltls):
        """process a batch of BLTL formulas"""
        verified_bltls = []
        
        for bltl in bltls:
            # remove unqualified BLTLs 
            attempts = 0
            while attempts < 2:
                refined_bltl = bltl 
                verification_result = self.verify_bltl(refined_bltl)
                if verification_result['valid']:
                    verified_bltls.append(bltl)
                    break
                else: 
                    print(f"Refining BLTL, attempt {attempts + 1}")
                    refined_bltl = self.refine_bltl(bltl, verification_result['error'])
                    attempts += 1

                    if not refined_bltl:
                        break
            
        return verified_bltls


