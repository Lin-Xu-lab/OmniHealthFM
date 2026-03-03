"""
step3_extract_llama_embeddings.py

What this script does
---------------------
This script extracts gene embeddings from a LoRA fine-tuned LLaMA model (Step 2).
It reads a gene list from ../data/gene_list.csv (single column named "gene"),
builds a short prompt for each gene, and computes embeddings using the last-layer
hidden states with mean pooling across tokens.

Inputs (user must provide)
--------------------------
1) ../data/gene_list.csv
   - CSV with one column named "gene" containing gene symbols (one per row)

2) LoRA checkpoint directory from Step 2
   - Default path in this script:
     ./lora_output_gene_string_GOmodeling/checkpoint-11100

Outputs
-------
- Gene_Embedding.pkl
  A pickled torch tensor with shape [n_genes, hidden_dim].

Reproducibility notes
---------------------
- Results depend on the exact LoRA checkpoint and base model.
- Tokenization/model/library versions can affect embeddings.
- GPU/CPU floating-point behavior may introduce small numeric differences.

Example
-------
python step3_extract_llama_embeddings.py

Dependencies
------------
numpy
pandas
torch
transformers
peft
"""

# ---------------------------------------------------------
# Load gene list from CSV (single column named "gene")
# ---------------------------------------------------------
import pickle
import numpy as np
import pandas as pd
import os

gene_list_path = "../data/gene_list.csv"
gene_df = pd.read_csv(gene_list_path)
gene_list = gene_df["gene"].astype(str).tolist()


# ---------------------------------------------------------
# Load LoRA fine-tuned Llama model (from Step 2)
# ---------------------------------------------------------
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

dev_num = 0
cuda = torch.cuda.is_available()
device = torch.device(f'cuda:{dev_num}' if cuda else 'cpu')

peft_model_path = "./lora_output_gene_string_GOmodeling/checkpoint-checkpoint-11100"  # Update if needed
config = PeftConfig.from_pretrained(peft_model_path)

# load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(
    config.base_model_name_or_path,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map={"": dev_num},  # force mapping to the selected GPU
    trust_remote_code=True
)

llama = PeftModel.from_pretrained(model, peft_model_path)
llama.to(device)
llama.eval()


# ---------------------------------------------------------
# Helper: compute gene embedding from a prompt
# Embedding = mean pooling of last hidden state across tokens
# ---------------------------------------------------------
def get_gene_embedding(gene_name):
    prompt = f"The function of gene {gene_name} is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = llama(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # [1, seq_len, hidden_dim]
        embedding = hidden_states.mean(dim=1)      # [1, hidden_dim]

    return embedding.squeeze(0).detach().cpu()


# ---------------------------------------------------------
# Extract embeddings for all genes and save
# ---------------------------------------------------------
E = torch.stack([get_gene_embedding(gene) for gene in gene_list])  # [n_genes, d]
gene_embeddings = E.detach().cpu()

with open("Gene_Embedding.pkl", "wb") as f:
    pickle.dump(gene_embeddings, f)