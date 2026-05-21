# BioBLTL
(Natural Language to Bounded Linear Temporal Logic Generation for Systems Biology)

BioBLTL is a framework for transforming natural-language descriptions of
biological temporal behaviors into Bounded Linear Temporal Logic (BLTL)
formulas using large language models, combining synthetic data generation,
Chain-of-Thought (CoT) preprocessing, and grammar-constrained decoding (GCD).

## Fine-tuning

We fine-tune LLMs (e.g., Qwen2.5, DeepSeek, Llama3, BioMistral) with LoRA
using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Ready-to-use
training and inference templates are provided under `training/`
(`train_lora_template.yaml` and `test_lora_template.yaml`); fill in
`dataset_dir`, `output_dir`, and `adapter_name_or_path`, register your dataset
in LLaMA-Factory's `dataset_info.json`, then run:

```bash
llamafactory-cli train training/train_lora_template.yaml   # fine-tune
llamafactory-cli train training/test_lora_template.yaml    # predict
```

Synthetic NL–BLTL training data can be generated from scratch with
`data_creation/prepare_syn_data.sh` (requires `OPENAI_API_KEY`). To use your
own data, format it as JSON pairs of `{"input": <NL>, "output": <BLTL>}`; 

At inference time, apply CoT preprocessing followed by grammar-constrained
decoding:

```bash
python gpt_preprocessor.py   # CoT-based NL rewriting
python test_gcd.py           # grammar-constrained BLTL generation
```

After inference, use `evaluate_llm.py` to report BLEU, Exact Match, and
Validity. For the model-checking module used in syntax verification and
DAG-equivalence evaluation, please contact the lab administrator.

## Citation

Difei Tang, Natasa Miskov-Zivanov, "Generating Bounded Linear Temporal Logic
in Systems Biology with Large Language Models", *bioRxiv*, 2025,
<https://www.biorxiv.org/content/10.1101/2025.08.06.668950v1.abstract>

## Funding

This work was funded in part by the NSF EAGER award CCF-2324742.

## Support

This research was supported in part by the University of Pittsburgh Center for
Research Computing, RRID:SCR_022735, through the resources provided.
Specifically, this work used the H2P cluster, which is supported by NSF award
number OAC-2117681.
