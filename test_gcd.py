from sympy import N
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
from pathlib import Path
from nltk.tokenize import word_tokenize
import json 
from tqdm import tqdm
import os 
import time

gcd_prompt = """Transform a sentence into a Bounded Linear Temporal Logic (BLTL) formula.\n
In this task, you will be provided with a sentence with a draft prediction that represents a generated BLTL formula. Your role is to
refine this draft prediction to ensure completeness and accuracy."""

def extract_word_spans_nltk(text):
    tokens = word_tokenize(text)

    words = [token for token in tokens if token.isalpha() or (token.isalnum() and not token.isdigit())]
    return words

def quote_term(s: str) -> str:
    s = s.replace('"', '\\"')
    return f'"{s}"'

def get_grammar_str(sentence, constraints=None):
    word = None 
    if not constraints:
        word_list = extract_word_spans_nltk(sentence)
        word = ' | '.join([quote_term(word) for word in word_list if word.strip() != ""])

        print('constraints:', word)

    # simple_grammar_str = f"""
    #     root ::= unit*
    #     unit ::= word | rel_op | logical_op | temporal_op | bound | others
    #     temporal_op ::= "F" | "G" | "U"
    #     rel_op ::= "=="
    #     logical_op ::= " & " | " | "
    #     bound ::= non_zero_digit digit*
    #     digit ::= [0-9]
    #     non_zero_digit ::= [1-9]
    #     others ::= "[" | "]" | "(" | ")"
    #     word ::= {word}
    #     """

    if constraints:
        variables = ' | '.join(constraints['variables'])
        time_bounds = ' | '.join(constraints['time_bounds'])

        print('constraints:', variables, time_bounds)

        grammar_str = f"""
            root ::= nested_formula

            nested_formula ::= formula (logical_op formula)*
            formula ::= unit ("U[" bound "]" unit)?
            unit ::= atomic | temporal_exp 
            temporal_exp ::= temporal "(" nested_formula ")"

            temporal ::= "G[" bound "]" | "F[" bound "]"
            atomic ::= variable rel_op value
            rel_op ::= "=="
            logical_op ::= "&"
            variable ::= {variables}
            value ::= "0" | "1"
            bound ::= {time_bounds}
            """
    else:
        grammar_str = f"""
            root ::= nested_formula

            nested_formula ::= formula (logical_op formula)*
            formula ::= unit ("U[" bound "]" unit)?
            unit ::= atomic | temporal_exp 
            temporal_exp ::= temporal "(" nested_formula ")"

            temporal ::= "G[" bound "]" | "F[" bound "]"
            atomic ::= variable rel_op value
            rel_op ::= "=="
            logical_op ::= "&"
            variable ::= {word}
            value ::= "0" | "1"
            bound ::= non_zero_digit digit*
            digit ::= [0-9]
            non_zero_digit ::= [1-9]
            """

    return grammar_str

def test_gcd(sentence, tokenizer, model, data_record=None):
    # Define prompts
    # Build system prompt
    system_prompt = """You are an expert in transforming the natural languages into Bounded Linear Temporal Logic (BLTL).
    The formula contains temporal operators with the following meanings:
    - F[t] φ: Finally: φ eventually has to hold (some where on the subsequent path).
    - G[t] φ: Globally: φ has to hold on the entire subsequent path.
    - ρ U[t] φ: Until: ρ has to hold at least until φ, which holds at the current or a future state.
    """

    instruction = None 
    constraints = None
    if data_record: 
        # TODO: use gcd as refinement 
        #instruction = data_record['instruction']
        instruction = gcd_prompt
        if 'variables' in data_record and 'time_bounds' in data_record:
            constraints = {
                'variables': map(str, data_record['variables']),
                'time_bounds': map(str, data_record['time_bounds']),
            }
    else:
        instruction = gcd_prompt

    grammar_str = get_grammar_str(sentence, constraints=constraints)

    # Create grammar constraint and logits processor
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

    user_prompt = (
        f"{instruction}\n"
        f"{sentence}"
    )


    messages = [
        #{"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # # Let HF handle device placement under device_map="auto"
    # inputs = tok([prompt], return_tensors="pt")

    # # Tokenize prompts
    # input_ids = tokenizer(example_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    # prompt_len = input_ids.shape[1]

    # Generate constrained text
    outputs = model.generate(
        **inputs,
        max_new_tokens=100, 
        logits_processor=[grammar_processor],
        repetition_penalty=1.1,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    predicted = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print(predicted)

    # generated_ids = output[0, prompt_len:]  # keep only continuation tokens
    # generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    # print(generated_text)

    return predicted 

def main():
    test_single_data = {
        'instruction': None,
        'input': 'Within 1400 time units, the concentration of RAS in PCCs will eventually reach 1, and then stay at this high level for 10000 time units starting from that point.',
        'output': 'F[5](FOXP3==1)'
    }

    nls = [test_single_data]

    #test_data = 'data/gold/preprocessed-vars-bounds-gcd/tcell_bltl_v2.json'
    test_data = 'data/gold/preprocessed-vars-bounds-gcd/pcc_bltl_v2.json'

    with open(test_data, 'r') as f:
        nls = [json.loads(line) for line in f]

    model_checkpoint = "Qwen/Qwen2.5-7B-Instruct"
    adapter_path = "<path/to/trained/adapter>/Qwen2.5-7B/lora/sft"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, adapter_path).eval() 
        
    output_name = 'generated_predictions.jsonl'
    output_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "output" / "gcd" / "pcc_1000"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / output_name

    generated_examples = []
    for i, data in tqdm(enumerate(nls), desc="Generating BLTL from natural language"):
        try:
            instruction = data['instruction']
            input_text = data['input']
            label = data['output']

            predicted = test_gcd(input_text, tokenizer=tokenizer, model=model, data_record=data)

            result_dict = {
                'prompt': input_text,
                "predict": predicted,
                "label": label,
            }
            
            generated_examples.append(result_dict)
            
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result_dict, ensure_ascii=False) + '\n')
            
            if i % 10 == 0:
                time.sleep(random.uniform(0.5, 2.0))
                
        except Exception as e:
            print(f"Error generating example {i}: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()