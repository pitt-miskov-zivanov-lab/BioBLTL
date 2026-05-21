"""
author: Difei Tang
email: DIT18@pitt.edu
MeLoDy Lab, University of Pittsburgh
"""

import json
from pathlib import Path
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import time
import pydantic
import os
from tqdm import tqdm
class RephraseNL(pydantic.BaseModel):
    NL: str

def enrich_dataset(input_file):
    """Enrich the existing synthetic dataset with rephrased descriptions
    """

    system_prompt = """You are an expert at rephrasing temporal logic descriptions.
    Your task is to rephrase the given description in a different way while:
    1. Maintaining the exact same temporal logic meaning
    2. Preserving all timing constraints (numbers)
    3. Keeping the same variable names (e.g., A, B, C)
    4. Using different but equivalent expressions

    Try you best to enrich the natural language descriptions like the following:
    1. Rephrasing the description "time units" to "time steps"
    2. Add more descriptions like the level, value expression, and concentration of the variable
    3. Add biological relationship descriptions:
    - A activates B (when A becomes 1, leading to B becoming 1)
    - A inhibits B (when A becomes 1, causing B to become 0)
    - A regulates B (when A's level changes, affecting B's level)
    4. Use diverse ways to describe values:
    - "becomes value 1" / "is activated"
    - "reaches value 0" / "is inhibited"
    - "maintains level between 0.3 and 0.7"
    5. Use diverse terms for temporal relationships:
    - "triggers"
    - "leads to"
    - "causes"
    - "results in"
    """
    
    user_prompt = """Please rephrase the following description in a different way.
    Keep the exact same meaning but vary the expression style.
    
    Original: {nl}
    
    Provide just one rephrased version.
    NL: 
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt)
    ])

    api_key = os.getenv("OPENAI_API_KEY")
    model = "gpt-4o-mini"
    llm = ChatOpenAI(
        api_key=api_key,
        model=model,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    structured_model = llm.with_structured_output(RephraseNL)
    chain = prompt | structured_model
    
    # Load existing data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    original_length = len(data)

    enriched_data = []
    output_file = "data/syn_enriched1.json"
    with open(output_file, 'a', encoding='utf-8') as f:
        for i in tqdm(range(original_length)):
            example = data[i]

            instruction = example['instruction']
            original_nl = example['input']
            bltl = example['output']
            
            try:
                result = chain.invoke({"nl": original_nl})
                
                # Add rephrased versions to data
                enriched_data = {
                    'instruction': instruction,
                    'input': result.NL,
                    'output': bltl,
                }

                f.write(json.dumps(enriched_data, ensure_ascii=False) + '\n')
                f.flush()
                
                # Add small delay to avoid rate limits
                if i % 5 == 0:
                    time.sleep(1)
                    
            except Exception as e:
                print(f"Error processing example {i}: {e}")
                continue
            
    print(f"enriched examples: {len(enriched_data)}")
    f.close()

if __name__ == "__main__":
    input_file = "data/syn.json"
    
    enrich_dataset(input_file)