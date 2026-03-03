"""
step2_finetune_llama.py

What this script does
---------------------
This script fine-tunes a pretrained Llama3 model on gene-level textual
knowledge using LoRA (Low-Rank Adaptation).

The input training data are JSONL files generated from Step 1
(step1_build_gene_knowledge.py), where each example corresponds to a
gene-level text description combining STRING interactions and GO terms.

Fine-tuning is performed using:
- 4-bit quantization (QLoRA-style) via bitsandbytes
- LoRA adapters applied to attention projection layers
- Hugging Face Trainer API

Inputs (user must provide)
--------------------------
- gene_string_GO_modeling-train.jsonl
- gene_string_GO_modeling-val.jsonl

Each JSONL entry should contain:
{
  "text": "gene: TP53 TP53 interacts with ... Associated GO terms: ..."
}

Outputs
-------
- LoRA adapter checkpoints saved to:
  ./lora_output_gene_string_GOmodeling/

These adapters are later used to extract gene embeddings (Step 3).

Reproducibility notes
---------------------
- This script requires access permission to the Meta LLaMA-3 model on Hugging Face.
- Model initialization depends on GPU availability and CUDA configuration.
- Training behavior may vary slightly across hardware and software environments.

Dependencies
------------
torch
transformers
datasets
peft
bitsandbytes
"""

# ---------------------------------------------------------
# Device and CUDA setup
# ---------------------------------------------------------
import torch

dev_num = 0
cuda = torch.cuda.is_available()
print('Is GPU available? ' + str(cuda))

device = torch.device(f'cuda:{dev_num}' if cuda else 'cpu')


# ---------------------------------------------------------
# Hugging Face & PEFT imports
# ---------------------------------------------------------
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import BitsAndBytesConfig


# ---------------------------------------------------------
# Load pretrained Llama-3 model and tokenizer
# NOTE: You must request access to LLaMA-3 on Hugging Face first
# ---------------------------------------------------------
model_id = "meta-llama/Meta-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token


# ---------------------------------------------------------
# Quantization configuration (4-bit, QLoRA-style)
# ---------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)


# ---------------------------------------------------------
# LoRA configuration
# ---------------------------------------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)


# ---------------------------------------------------------
# Load training and validation datasets (JSONL)
# ---------------------------------------------------------
data = load_dataset(
    "json",
    data_files={
        "train": "./data/gene_string_GO_modeling-train.jsonl",
        "validation": "./data/gene_string_GO_modeling-val.jsonl"
    }
)


# ---------------------------------------------------------
# Tokenization
# ---------------------------------------------------------
def tokenize(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

tokenized_dataset = data.map(tokenize, batched=True)


# ---------------------------------------------------------
# Data collator for causal language modeling
# ---------------------------------------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


# ---------------------------------------------------------
# Training configuration
# ---------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./lora_output_gene_string_GOmodeling",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=20,
    fp16=True,
    report_to="none"
)


# ---------------------------------------------------------
# Trainer setup and training
# ---------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()