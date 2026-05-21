"""
author: Difei Tang
email: DIT18@pitt.edu
MeLoDy Lab, University of Pittsburgh
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import random
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import time
import pydantic
from DiverseDA import DiverseDA
from dotenv import load_dotenv

from datetime import datetime
current_date = datetime.now().strftime("%Y%m%d")

load_dotenv()

class BLTLNL(pydantic.BaseModel):
    NL: str

def generate_bltl_nl_pairs(
    bltls,
    output_name="generated_data_{current_date}", 
    samples = [],
    use_diverse = False,
    seq_var = None,
):
    """
    Use LangChain and ChatGPT to generate BLTL and natural language description pairs
    
    Args:
        bltls: List of BLTL properties
        api_key: OpenAI API key
        model: Model name
        output_file: Output file name
    """

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("LLM_MODEL")

    # Create ChatGPT model instance
    llm = ChatOpenAI(model_name=model, temperature=0.7)
    
    # Build system prompt
    system_prompt = """You are an expert in transforming the Bounded Linear Temporal Logic (BLTL) into natural language (NL) descriptions.
    The formula contains temporal operators with the following meanings:
    - F[t] φ: Finally: φ eventually has to hold (some where on the subsequent path).
    - G[t] φ: Globally: φ has to hold on the entire subsequent path.
    - ρ U[t] φ: Until: ρ has to hold at least until φ, which holds at the current or a future state.

    """

    task_description_prompt = ("Below is an instruction that describes a task. Please generate appropriate content as required.\n"
                           "Task: Translate Bounded Linear Temporal Logic (BLTL) formulas to natural language (NL) descriptions.\n"
                           "Definition: BLTL formulas describe temporal properties of biological systems using logical operators and time bounds. "
                           "Your task is to convert these formal logical expressions into clear, natural language descriptions that explain what the formula means in biological terms.\n")

    re_prompt = ("Here is a brief description of Bounded Linear Temporal Logic (BLTL) formula:\n"
                 "The formula contains temporal operators with the following meanings:\n"
                 "- F[t] φ: Finally: φ eventually has to hold (somewhere on the subsequent path within t time units).\n"
                 "- G[t] φ: Globally: φ has to hold on the entire subsequent path for t time units.\n"
                 "- ρ U[t] φ: Until: ρ has to hold at least until φ, which holds at the current or a future state within t time units.\n"
                 "The formula contains logical operators with the following meanings:\n"
                 "- & (AND): Both conditions must be true simultaneously.\n"
                 "- | (OR): At least one condition must be true.\n"
                 "- ==, >, <, >=, <=: Comparison operators for numerical values.\n")
    
    # TODO: 
    if use_diverse:
        if seq_var:
            suffix_prompt = ("Now transform the following BLTL formula to natural language. Please make the generated samples as different from the above demonstration as possible.\n"
"This formula contains SEQUENTIAL BEHAVIOR for variable(s): {seq_var}. "
"The variable(s) change state over multiple time points, creating a temporal sequence. "

"Generate natural language that emphasizes:\n"
"- Temporal progression: Use expressions like 'becomes', 'changes to', 'transitions from X to Y', 'initially', 'then', 'finally'\n"
"- Time relationships: 'by round X', 'until round Y', 'from round A to round B', 'lasting for Z rounds'\n"
"- Sequential transitions: Show how {seq_var} evolves over time with clear temporal markers\n"
"- State changes: Explicitly mention when {seq_var} switches between values (0→1, 1→0)\n"

"Example patterns to use:\n"
"- '{seq_var} becomes 1 by round X, but changes back to 0 by round Y'\n"
"- '{seq_var} initially activates, then later deactivates'\n"
"- '{seq_var} transitions from 0 to 1, and after Z rounds, returns to 0'\n"
"- 'By round X, {seq_var} reaches value 1, and by round Y (lasting until round Z), {seq_var} becomes 0'\n"

"Avoid repetitive language and create varied, natural expressions.\n"
                            "BLTL:{bltl_formula}\n"
                            "NL:\n")
        else:
            suffix_prompt = ("Now transform the following BLTL formula to natural language. Please make the generated samples as different from the above demonstration as possible.\n"
                            "Avoid repetitive language and create varied, natural expressions.\n"
                            "BLTL:{bltl_formula}\n"
                            "NL:\n")
    else:
        if seq_var:
            suffix_prompt = ("Now transform the following BLTL formula to natural language:\n"
"This formula contains SEQUENTIAL BEHAVIOR for variable(s): {seq_var}. "
"The variable(s) change state over multiple time points, creating a temporal sequence. "

"Generate natural language that emphasizes:\n"
"- Temporal progression: Use expressions like 'becomes', 'changes to', 'transitions from X to Y', 'initially', 'then', 'finally'\n"
"- Time relationships: 'by round X', 'until round Y', 'from round A to round B', 'lasting for Z rounds'\n"
"- Sequential transitions: Show how {seq_var} evolves over time with clear temporal markers\n"
"- State changes: Explicitly mention when {seq_var} switches between values (0→1, 1→0)\n"

"Example patterns to use:\n"
"- '{seq_var} becomes 1 by round X, but changes back to 0 by round Y'\n"
"- '{seq_var} initially activates, then later deactivates'\n"
"- '{seq_var} transitions from 0 to 1, and after Z rounds, returns to 0'\n"
"- 'By round X, {seq_var} reaches value 1, and by round Y (lasting until round Z), {seq_var} becomes 0'\n"

"Avoid repetitive language and create varied, natural expressions.\n"
                            "BLTL:{bltl_formula}\n"
                            "NL:\n")
        else:
            suffix_prompt = ("Now transform the following BLTL formula to natural language:\n"
                        "Avoid repetitive language and create varied, natural expressions.\n"
                        "BLTL:{bltl_formula}\n"
                        "NL:\n")
   
    dda = DiverseDA(samples=samples)
    dda.update_task_description(task_description_prompt)
    dda.update_explanation(re_prompt)

    # build the prompt 
    dda.build_prompt(suffix_prompt=suffix_prompt)
    user_prompt = dda.prompt
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
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

    structured_model = llm.with_structured_output(BLTLNL)
    chain = prompt | structured_model

    output_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "data" / "generated" / "ori"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{output_name}.json'

    generated_examples = []

    for i, bltl in tqdm(enumerate(bltls), desc="Generating BLTL2NL pairs"):
        try:
            # FIXME: seems that seq_var has no effect on the generation of NL
            if seq_var:
                result = chain.invoke({"bltl_formula": bltl, "seq_var": seq_var})
            else:
                result = chain.invoke({"bltl_formula": bltl})
            
            result_dict = {
                "BLTL": bltl,
                "NL": result.NL,
                "Type": "Generated",
                "Index": f"Generated_{i}",
                "Test Hypotheses": "None",
                "id": i
            }
            
            generated_examples.append(result_dict)
            
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result_dict, ensure_ascii=False) + '\n')
            
            # renew few-shot examples
            if use_diverse:
                generated_bltl = result_dict['BLTL']
                generated_nl = result_dict['NL']
                dda.add_sample(f'BLTL:{generated_bltl}' + '\n' + f'NL:{generated_nl}')

            if i % 10 == 0:
                time.sleep(random.uniform(0.5, 1.0))
                
        except Exception as e:
            print(f"Error generating example {i}: {e}")
            time.sleep(5)

    return generated_examples

def create_bltl_nl_samples():
    root_dir = os.path.dirname(os.path.dirname(__file__)) 
    samples_file = os.path.join(root_dir, 'data/gold/curated_samples_seq.xlsx')
    samples_df = pd.read_excel(samples_file)

    samples_bltls = samples_df['BLTL'].to_list()
    samples_nls = samples_df['NL'].to_list()

    samples = []
    for bltl, nl in zip(samples_bltls, samples_nls):
        sample = f'BLTL:{bltl}' + '\n' + f'NL:{nl}'
        samples.append(sample)
    
    return samples

def main():
    # samples = create_bltl_nl_samples()

    # bltls = ['F[37](G[30](mTOR == 1.0)) | F[25](G[3](erlotinib == 0.0 & p70S6K == 0.4))']

    # generated_examples = generate_bltl_nl_pairs(
    #     bltls=bltls, 
    #     output_name="generated_data.jsonl",
    #     samples=samples
    # )

    #print(generated_examples)
    pass 

if __name__ == "__main__":
    main()

    