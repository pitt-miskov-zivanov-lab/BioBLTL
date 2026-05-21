"""
@author: Difei Tang 
@email: DIT18@pitt.edu

Credits to: https://arxiv.org/abs/2505.23108
"""
import re
import json
import openai
import os 
from dotenv import load_dotenv

# TODO: adding more functionalities 
class DiverseDA():
    def __init__(self, max_samples=5, samples=[]):
        self.tdp = "" # task description module  
        self.ep = "" # explanation module
        self.samples = samples # samples used in ICL  
        self.max_samples = max_samples
        self.sdp = "" # sample demonstration module 

        if len(self.samples) > 0:
            self.update_demonstrations()
        else:
            print("Note: sample demonstrations are not empty")

        self.sp = None
        self.prompt = ""

    def add_sample(self, sample):
        """Add a new sample to the demonstrations"""
        self.samples.append(sample)
        if len(self.samples) > self.max_samples:
            self.samples.pop(0)

        self.update_demonstrations()
    
    def add_samples(self, samples):
        for sample in samples: 
            self.samples.append(sample)
            if len(self.samples) > self.max_samples:
                self.samples.pop(0)

        self.update_demonstrations()

    def update_demonstrations(self):
        """Update the sample demonstrations"""

        examples_text = "Here are some examples:\n"

        for i, sample in enumerate(self.samples, 1):
            examples_text += f"{sample}\n"
        
        self.sdp = examples_text

    def update_task_description(self, input_str: str):
        self.tdp = input_str

    def update_explanation(self, input_str: str):
        self.ep = input_str

    def add_prompt(self, input_str: str):
        self.sp = input_str

    def build_prompt(self, suffix_prompt: str):
        if suffix_prompt: 
            self.add_prompt(suffix_prompt)

        if self.sp:
            self.prompt = self.tdp + \
                    self.ep + \
                    self.sdp + \
                    self.sp
        else: 
            raise ValueError(f"main prompt is None")
    
        return self.prompt

def test_diverseDA():
    """A test function: prompting LLMs for biomedical relation extraction"""
    task_description_prompt = (f"Below is an instruction that describes a task. Please genearte appropriate content as required.\n"
                           f"Definition: One sample in relation extraction datasets consists of a relation, a context and a pair of head and tail entities in the context, and their location information.\n")

    re_prompt = (f"Here is a brief explanation of a relationship type in DDIs (drug-drug interactions): \n"
                f"DDI-mechanism: This type is used to annotate DDIs that are described by their PK mechanism (e.g. Grepafloxacin may inhibit the metabolism of theobromine).\n"\
                f"DDI-effect: This type is used to annotate DDIs describing an effect (e.g. In uninfected volunteers, 46% developed rash while receiving SUSTIVA and clarithromycin) or a PD mechanism (e.g. Chlorthalidone may potentiate the action of other antihypertensive drugs).\n"
                f"DDI-advise: This type is used when a recommendation or advice regarding a drug interaction is given (e.g. UROXATRAL should not be used in combination with other alpha-blockers).\n"
                f"DDI-int: This type is used when a DDI appears in the text without providing any additional information (e.g. The interaction of omeprazole and ketoconazole has been established).\n"
                f"None: This type is used when there is no DDI apprears in the text\n")
    
    # fake data 
    samples = ['1', '2', '3']

    rtype = 'disease'
    suffix_prompt = (f"So please generate a sample for the relation '{rtype}'. Please make the generated samples as different from the above demonstration as possible.\n\n"
                    f"###Response:\n")

    # initialize the diverse DA 
    dda = DiverseDA(samples=samples)
    dda.update_task_description(task_description_prompt)
    dda.update_explanation(re_prompt)
    # build the prompt 
    dda.build_prompt(suffix_prompt=suffix_prompt)

    print(dda.prompt)

def main():
    #test_diverseDA()
    pass 
    
if __name__ == "__main__":
    main()