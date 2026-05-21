"""
GPT-based Preprocessor for natural language describing temporal logics 
author: Difei Tang
email: DIT18@pitt.edu
MeLoDy Lab, University of Pittsburgh
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import pydantic
import json
import os
import pandas as pd 
from typing import List, Optional
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

class PreprocessorResult(pydantic.BaseModel):
    HAS_SEQUENTIAL_BEHAVIOR: bool
    REASONING1: str
    NEEDS_REPHRASING: bool
    REASONING2: str
    REPHRASED_SENTENCE: str
    VARIABLES: List[str]
    TIME_BOUNDS: List[int]

def create_gpt_preprocessor():
    """
    Create a GPT-based preprocessor for natural language sentences.
    
    Returns:
        A function that can preprocess natural language sentences
    """
    
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("LLM_MODEL")
    
    # Build preprocessing prompt
#     preprocessing_prompt_v0 = """You are an expert preprocessor for converting natural language descriptions of biological behaviors into Bounded Linear Temporal Logic (BLTL).

# Your task is to analyze the given sentence and:
# 1. Extract variables: Biological entities that can have values (e.g., IL2, P53, apoptosis, VEGF)
# 2. Extract time bounds: Numerical time references (e.g., 10 from "by round 10", 20 from "within 20 time units")
# 3. Detect sequential behavior: Whether variables change values over time in sequence
# 4. Detect state descriptions: Whether the sentence describes a specific state with multiple variables and their values
# 5. Determine if rephrasing is needed to make temporal relationships clearer
# 6. If rephrasing is needed, provide a clearer version
# 7. Make sure the time bounds are consistent with the rephrased sentence

# Key rephrasing rules:
# - For sequential behavior: Convert "X becomes 1 by round 10, but changes back to 0 by round 30" 
#   → "X becomes 1 by round 10, and then changes back to 0 within 20 rounds after that" (calculate time difference: 30-10=20)
# - For state descriptions: Remove confusing state references like "State A11" and make the sentence more direct
#   "State A11, where X are 0 and Y are 1, must occur at least once within 20 rounds"
#   → "Within 20 rounds, X must be 0 and Y must be 1, at least once"
# - For global context: Remove contextual references that are not variables, such as "in PCCs", "in PSCs", "in the tumor microenvironment", "in the cell", etc.
#   "In PCCs, within 1000 time units, proliferation will eventually be activated to 1"
#   → "Within 1000 time units, proliferation will eventually be activated to 1"

# Examples:

# INPUT: "IL-2 becomes 1 by round 10, but changes back to 0 until round 30"
# OUTPUT:
# - has_sequential_behavior: true
# - has_state_description: false
# - needs_rephrasing: true
# - rephrased_sentence: "IL2 becomes 1 by round 10, and then changes back to 0 within 20 rounds after that"
# - variables: ["IL2"]
# - time_bounds: [10, 20]
# - explanation: "Sequential behavior detected. Calculated time difference (30-10=20) and rephrased to make temporal sequence clearer."

# INPUT: "State A11, where IL2, AKT, and MTORC1 are 0, and FOXP3, PTEN, RAS, P13K, CD25 and MTORC2 are 1, must occur at least once within the first 20 rounds"
# OUTPUT:
# - has_sequential_behavior: false
# - has_state_description: true
# - needs_rephrasing: true
# - rephrased_sentence: "Within the first 20 rounds, IL2, AKT, and MTORC1 must be 0, and FOXP3, PTEN, RAS, P13K, CD25 and MTORC2 must be 1, at least once"
# - variables: ["IL2", "AKT", "MTORC1", "FOXP3", "PTEN", "RAS", "P13K", "CD25", "MTORC2"]
# - time_bounds: [20]
# - explanation: "State description detected. Removed confusing 'State A11' reference and restructured sentence for clarity."

# INPUT: "MTORC1 becomes 1 by round 9, IL2 becomes 1 by round 10, IL2_EX becomes 1 by round 12, FOXP3 becomes 1 by round 13, and PTEN becomes 1 by round 15, and by round 20 (and until round 30) FOXP3=0, IL2=0, MTORC1=0, PTEN=1, CD25=0, MTORC2=0."
# OUTPUT:
# - has_sequential_behavior: true
# - has_state_description: false
# - needs_rephrasing: true
# - rephrased_sentence: "MTORC1 becomes 1 by round 9, then IL2 becomes 1 by round 10, then IL2_EX becomes 1 by round 12, then FOXP3 becomes 1 by round 13, then PTEN becomes 1 by round 15, and finally by round 20 and last for 10 rounds, FOXP3, IL2, MTORC1, CD25, and MTORC2 must be 0 while PTEN must be 1"
# - variables: ["MTORC1", "IL2", "IL2_EX", "FOXP3", "PTEN", "CD25", "MTORC2"]
# - time_bounds: [9, 10, 12, 13, 15, 20, 10]
# - explanation: "Complex sequential behavior detected (until round 30) with multiple variables changing at different time points and a final state lasting for a duration. Calculated duration (30-20=10) and rephrased to clarify the temporal sequence and final state conditions."

# Now analyze this sentence:
# INPUT: "{sentence}"
# OUTPUT:"""

    preprocessing_prompt_v1 = """You are an expert preprocessor for converting natural language descriptions of biological behaviors into Bounded Linear Temporal Logic (BLTL).

Your task is to analyze the given sentence and:
1. Extract variables: Biological entities that can have values (e.g., IL2, P53, apoptosis, VEGF)
2. Extract time bounds: Numerical time references (e.g., 10 from "by round 10", 20 from "within 20 time units")
3. Detect sequential behavior: Examine each variable individually to see if it changes values over time in sequence
4. Check rephrasing needs: For each variable with sequential behavior, determine if the temporal relationship needs clarification
5. Rephrase if needed: Convert ambiguous temporal patterns to clear "by round X, then within Y rounds" format
6. Provide reasoning: Explain your detection logic and rephrasing decisions
7. Ensure consistency: Make sure time bounds match the rephrased sentence
8. If you detect sequential behavior in ANY variable, you MUST rephrase that variable's temporal description, regardless of other sentence complexity. Do not skip rephrasing just because the sentence contains other patterns.

Example1:
INPUT: 
IL2 becomes 1 by round 10, but changes back to 0 until round 30.
OUTPUT:
HAS_SEQUENTIAL_BEHAVIOR: true
REASONING1: IL2 transitions from 0 to 1, then back to 0, showing sequential behavior with multiple state changes.
NEEDS_REPHRASING: true
REASONING2: The phrase "changes back to 0 until round 30" is ambiguous. It should be "changes back to 0 within 20 rounds after that" to clarify the temporal sequence. Calculation: 30 - 10 = 20.
REPHRASED_SENTENCE: IL2 becomes 1 by round 10, and then changes back to 0 within 20 rounds after that
VARIABLES: IL2
TIME_BOUNDS: 10, 20

Example2:
INPUT: 
MTORC1 becomes 1 by round 9, IL2 becomes 1 by round 10, IL2_EX becomes 1 by round 12, FOXP3 becomes 1 by round 13, and PTEN becomes 1 by round 15, and by round 20 (and until round 30) FOXP3=0, IL2=0, MTORC1=0, PTEN=1, CD25=0, MTORC2=0.
OUTPUT:
HAS_SEQUENTIAL_BEHAVIOR: true
REASONING1: Multiple variables change states sequentially over time, and the final state "by round 20 (and until round 30)" indicates a duration pattern, showing sequential behavior.
NEEDS_REPHRASING: true
REASONING2: The phrase "by round 20 (and until round 30)" is confusing. It means the final state starts at round 20 and lasts for 10 rounds. Calculation: 30 - 20 = 10. The sentence needs restructuring to clarify the temporal sequence and final state duration.
REPHRASED_SENTENCE: MTORC1 becomes 1 by round 9, then IL2 becomes 1 by round 10, then IL2_EX becomes 1 by round 12, then FOXP3 becomes 1 by round 13, then PTEN becomes 1 by round 15, and finally by round 20 and last for 10 rounds, FOXP3, IL2, MTORC1, CD25, and MTORC2 must be 0 while PTEN must be 1
VARIABLES: MTORC1, IL2, IL2_EX, FOXP3, PTEN, CD25, MTORC2
TIME_BOUNDS: 9, 10, 12, 13, 15, 20, 10

Example3:
INPUT: 
Foxp3 equals 0 until round 5, then equals 1 until round 12, then equals 0 until round 15, then again equals 1 until round 30.
OUTPUT:
HAS_SEQUENTIAL_BEHAVIOR: true
REASONING1: Foxp3 transitions between states 0 and 1 multiple times over different time periods, showing complex sequential behavior with multiple state changes.
NEEDS_REPHRASING: true
REASONING2: The "until X, then until Y" pattern is ambiguous. Each transition needs to be clarified with specific start times and durations.
Foxp3 starts at 0 until round 5, then changes to 1 until round 12, so it becomes 1 by round 6 and lasts for 6 rounds. Time bound 6 is the time the value changes, and 12 - 6 = 6. 
Foxp3 then changes to 0 until round 15, so it becomes 0 by round 13 and lasts for 2 rounds. Time bound 13 (12 + 1) is the time the value changes, and 15 - 13 = 2.
Foxp3 then changes to 1 until round 30, so it becomes 1 by round 16 and lasts for 14 rounds. Time bound 16 (15 + 1) is the time the value changes, and 30 - 16 = 14.
REPHRASED_SENTENCE: Foxp3 equals 0 until round 5, then equals 1 by round 6 and lasts for 6 rounds, then equals 0 by round 13 and lasts for 2 rounds, and finally equals 1 by round 16 for 14 rounds
VARIABLES: FOXP3
TIME_BOUNDS: 5, 6, 6, 13, 2, 16, 14

Example5:
INPUT: 
Within 20 rounds, VEGF will eventually reach 1
OUTPUT:
HAS_SEQUENTIAL_BEHAVIOR: false
REASONING1: VEGF only reaches one state (1), so no sequential behavior.
NEEDS_REPHRASING: false
REASONING2: The sentence is already clear. No rephrasing needed.
REPHRASED_SENTENCE: "Within 20 rounds, VEGF will eventually reach 1"
VARIABLES: VEGF
TIME_BOUNDS: 20

Now analyze this sentence:
INPUT: "{sentence}"
OUTPUT:"""
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("user", preprocessing_prompt_v1)
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
    structured_model = llm.with_structured_output(PreprocessorResult)
    chain = prompt | structured_model
    
    def preprocess_sentence(sentence: str) -> PreprocessorResult:
        """
        Preprocess a natural language sentence using GPT.
        
        Args:
            sentence: Natural language description of biological behavior
            
        Returns:
            PreprocessorResult with detected metadata and rephrased sentence
        """
        try:
            result = chain.invoke({"sentence": sentence})
            return result
        except Exception as e:
            print(f"Error preprocessing sentence: {e}")
            # Fallback: return basic structure
            return PreprocessorResult(
                variables=[],
                time_bounds=[],
                has_sequential_behavior=False,
                has_state_description=False,
                needs_rephrasing=False,
                rephrased_sentence=sentence,
                explanation="Error occurred during preprocessing, returning original sentence."
            )
    
    return preprocess_sentence

def test_gpt_preprocessor():
    """
    Test the GPT preprocessor with example sentences.
    """
    preprocessor = create_gpt_preprocessor()
    
    test_sentences = [
        #"IL2 becomes 1 by round 10, but changes back to 0 by round 30",
        #"State A11, where IL2, AKT, and MTORC1 are 0, and FOXP3, PTEN, RAS, P13K, CD25 and MTORC2 are 1, must occur at least once within the first 20 rounds",
        #"Within 1000 time units, VEGF will eventually reach 1",
        "FOXP3 never becomes 1 until round 30, MTORC1 becomes 1 by round 5, CD25 becomes 1 by round 6, PI3K becomes 1 by round 3, then PI3K becomes 0 by round 7, and by round 20 (and until round 30) FOXP3=0, IL2=1, MTORC1=1, PTEN=0, CD25=1, MTORC2=1, PI3K=1, RAS=1",
        #"Steady state Foxp3=0, IL2=1, PTEN=0, CD25=1, PI3K=1, MTORC1=1 and MTORC2=1 is reached by round 5 (and until round 30)."
    ]
    
    for sentence in test_sentences:
        print(f"Original: {sentence}")
        result = preprocessor(sentence)
        
        print(f"Sequential behavior: {result.HAS_SEQUENTIAL_BEHAVIOR}")
        print(f"Reasoning1: {result.REASONING1}")
        print(f"Needs rephrasing: {result.NEEDS_REPHRASING}")
        print(f"Reasoning2: {result.REASONING2}")
        print(f"Rephrased: {result.REPHRASED_SENTENCE}")
        print(f"Variables: {result.VARIABLES}")
        print(f"Time bounds: {result.TIME_BOUNDS}")
        print("---")

def preprocess_biobltl_data_2():
    input_excel_files = ['data/gold/ori/Tcell_Natasa_curated.xlsx']
    #input_excel_files = ['data/gold/ori/Tcell_Natasa_curated.xlsx', 'data/gold/ori/PCC_Qinsi_curated.xlsx']

    preprocessor = create_gpt_preprocessor()

    for file in tqdm(input_excel_files):
        all_results = []
        df = pd.read_excel(file, usecols=['BLTL', 'NL'])

        for index, row in df.iterrows():
            bltl = row['BLTL']
            nl = row['NL']

            result = preprocessor(nl)
            if not result.NEEDS_REPHRASING:
                new_nl = nl
            else:
                new_nl = result.REPHRASED_SENTENCE

            record = {
                'BLTL': bltl,
                'NL': nl,
                'new_NL': new_nl,
                'has_sequential_behavior': result.HAS_SEQUENTIAL_BEHAVIOR,
                'reasoning1': result.REASONING1,
                'needs_rephrasing': result.NEEDS_REPHRASING,
                'reasoning2': result.REASONING2,
                'variables': result.VARIABLES,
                'time_bounds': result.TIME_BOUNDS,
            }

            all_results.append(record)

        results_df = pd.DataFrame(all_results)

        results_df.to_excel(file.replace('.xlsx', '_preprocessed.xlsx'), index=False)

if __name__ == "__main__":
    test_gpt_preprocessor()
    #preprocess_biobltl_data_2()
