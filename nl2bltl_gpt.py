"""
author: Difei Tang
email: DIT18@pitt.edu
MeLoDy Lab, University of Pittsburgh

A simple script to use LLMs to convert natural language descriptions into BLTLs 
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional
from pathlib import Path
import os
import pydantic
import json
import time
import random
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

class NLBLTL(pydantic.BaseModel):
    OUTPUT: str

def generate_bltl_gpt(nls, task: str, output_name: str = None):
    """
    Initialize the BLTL inferencer with ChatGPT.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("LLM_MODEL")

    # Build system prompt
    system_prompt = """You are an expert in transforming natural language (NL) to Bounded Linear Temporal Logic (BLTL).
    The formula contains temporal operators with the following meanings:
    - F[t] φ: Finally: φ eventually has to hold (some where on the subsequent path).
    - G[t] φ: Globally: φ has to hold on the entire subsequent path.
    - ρ U[t] φ: Until: ρ has to hold at least until φ, which holds at the current or a future state.
    """

    # system_prompt = """You are an expert in transforming natural language (NL) to Bounded Linear Temporal Logic (BLTL).
    # The formula contains temporal operators with the following meanings:
    # - F[t] φ: Finally: φ eventually has to hold (some where on the subsequent path).
    # - G[t] φ: Globally: φ has to hold on the entire subsequent path.
    # - ρ U[t] φ: Until: ρ has to hold at least until φ, which holds at the current or a future state.
    # Below are some BLTL formulas that we allow: 
    # F[20](A==1), G[15](A==1&B==0), F[25](G[10](A == 1)), (A == 1 & B == 0) U[30] (C == 1)
    # F[10](A==1&F[5](B==1)), F[10](A==0)&F[15](A==0)
    # """

    # Build user prompt w/o few-shot examples 
    user_prompt = """
    Here are some examples:\n
    INPUT: Within 20 time units, A will eventually become 1.
    OUTPUT: F[20](A == 1) \n
    INPUT: For the next 15 time units, A remains one, and B remains zero continuously.
    OUTPUT: G[15](A == 1 & B == 0) \n
    INPUT: Within 25 time units, A will eventually become 1 and then maintain this state for 10 time units.
    OUTPUT: F[25](G[10](A == 1)) \n
    INPUT: A remains 1 and B remains 0, and this state continues until C becomes 1 within 30 time units.
    OUTPUT: (A == 1 & B == 0) U[30] (C == 1) \n{instruction}
    INPUT: {input_text}
    OUTPUT:
    """

    # user_prompt = """{instruction}
    # INPUT: {input_text}
    # OUTPUT:
    # """
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt)
    ])

    llm = ChatOpenAI(
        api_key=api_key,
        model=model,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    
    # Create the chain
    structured_model = llm.with_structured_output(NLBLTL)
    chain = prompt | structured_model
    
    output_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "output" / "gpt-few-shot" / task
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / output_name

    generated_examples = []

    with open(output_file, 'a', encoding='utf-8') as f:
        for i, data in tqdm(enumerate(nls), desc="Generating BLTL from natural language"):
            try:
                instruction = data['instruction']
                input_text = data['input']
                label = data['output']

                result = chain.invoke({"instruction": instruction, "input_text": input_text})
                
                result_dict = {
                    # "prompt": user_prompt.format(
                    #     instruction=instruction,
                    #     input_text=input_text
                    # ),
                    "prompt": input_text,
                    "predict": result.OUTPUT,
                    "label": label,
                }
                
                generated_examples.append(result_dict)

                f.write(json.dumps(result_dict, ensure_ascii=False) + '\n')
                f.flush()  
                    
            except Exception as e:
                print(f"Error generating example {i}: {e}")
                time.sleep(5)

def main():
    # Example usage
    # nls = ["Within 20 time units, A will eventually become 1",
    #        "For the next 15 time units, A remains one, and B remains zero continuously",
    #        "Within 25 time units, A will eventually become 1 and then maintain this state for 10 time units",
    #        "A remains 1 and B remains 0, and this state continues until C becomes 1 within 30 time units"]

    # gold biobltl data
    # input_files = {
    # 'ori': 'data/gold/ori/pcc_tcell_bltl_ori.json',
    # 'v0': 'data/gold/preprocessed/pcc_tcell_bltl_v0.json',
    # 'v1': 'data/gold/preprocessed-vars-bounds/pcc_tcell_bltl_v1.json',
    # 'v2': 'data/gold/preprocessed-vars-bounds-gcd/pcc_tcell_bltl_v2.json'}

    # synthetic test data
    input_files = {
        'syn': 'data/generated/release/synbltl_5000/synbltl_test.json'
    }

    for task, input_file in input_files.items():
        with open(input_file, 'r') as f:
            nls = [json.loads(line) for line in f]
        generate_bltl_gpt(nls, task=task, output_name="generated_predictions.jsonl")

if __name__ == "__main__":
    main()