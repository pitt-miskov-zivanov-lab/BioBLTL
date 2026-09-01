<h1 align="center"><samp>NL2BLTL</samp></h1>

<p align="center">
  Automated Generation of Bounded Linear Temporal Logic<br>
  from Natural Language for Systems Biology
</p>


<p align="center">
  <a href="https://boheme.pitt.edu/nl2bltl"><strong>Live Demo</strong></a>
</p>
NL2BLTL is a framework for transforming natural-language descriptions of biological temporal behaviors into Bounded Linear Temporal Logic (BLTL) formulas using large language models (LLMs). It combines synthetic data generation, supervised fine-tuning, Chain-of-Thought (CoT) preprocessing, and grammar-constrained decoding (GCD).

## Example

```text
NL:   Within 20 rounds, FOXP3 eventually becomes active and remains active for 10 rounds.
BLTL: F[20](G[10](FOXP3 == 1))
```

Try NL2BLTL through the [live demo](https://boheme.pitt.edu/nl2bltl), or download the best-performing model checkpoint from [Hugging Face](https://huggingface.co/DifeiT/nl2bltl).

## Download and Run

The released checkpoint has been tested on an NVIDIA A100 GPU (40 GB VRAM) and an AMD CPU. Other hardware configurations have not been tested. A CUDA-capable GPU is recommended, and approximately 15 GB of disk space is required to download the model.

```bash
hf download DifeiT/nl2bltl --local-dir nl2bltl-model
```

```python
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="nl2bltl-model",
    device_map="auto",
    torch_dtype="auto",
)

sentence = (
    "Within 20 rounds, FOXP3 eventually becomes active and remains active "
    "for 10 rounds."
)
prompt = (
    "Transform the following sentence into a Bounded Linear Temporal Logic "
    f"(BLTL) formula.\n{sentence}"
)
result = generator(
    [{"role": "user", "content": prompt}],
    max_new_tokens=64,
)
print(result[0]["generated_text"][-1]["content"])
```

## Fine-Tuning

We fine-tune LLMs such as Qwen2.5, DeepSeek, Llama 3, and BioMistral with LoRA using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Training and inference templates are provided under `training/`. Update the paths and dataset names in the templates, register the datasets in LLaMA-Factory's `dataset_info.json`, and run:

```bash
llamafactory-cli train training/train_lora_template.yaml
llamafactory-cli train training/test_lora_template.yaml
```

## Data Generation and Inference

Synthetic NL-BLTL training data can be generated with:

```bash
cd data_creation
bash prepare_syn_data.sh
```

For CoT preprocessing and grammar-constrained decoding:

```bash
python gpt_preprocessor.py
python -m pip install transformers-cfg
python test_gcd.py
```

After inference, use `evaluate_llm.py` to report BLEU, exact/equivalence accuracy, and parser validity. 

## Citation

Difei Tang, Natasa Miskov-Zivanov, "Generating Bounded Linear Temporal Logic
in Systems Biology with Large Language Models", _bioRxiv_, 2025,
[https://www.biorxiv.org/content/10.1101/2025.08.06.668950v1.abstract](https://www.biorxiv.org/content/10.1101/2025.08.06.668950v1.abstract)

## Funding

This work was funded in part by the NSF EAGER award CCF-2324742.

## Support

This research was supported in part by the University of Pittsburgh Center for
Research Computing, RRID:SCR_022735, through the resources provided.
Specifically, this work used the H2P cluster, which is supported by NSF award
number OAC-2117681.
