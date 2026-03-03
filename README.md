# OmniHealthFM
Pan-Disease Prediction for Diagnosis, Response, and Prognosis with A Knowledge-Driven Transcriptomic Large Language Model

# Introduction

OmniHealthFM is a knowledge-guided foundation model trained on bulk transcriptomic data from 344,653 individuals. It integrates gene expression profiles with curated biological knowledge using Llama 3–derived gene embeddings and a cross-attention framework.

The model produces generalizable transcriptomic representations and outperforms existing methods across clinical tasks such as diagnosis, prognosis, and treatment response prediction.
![Fig](/Image/omni_figure.png) 

# Overview of the Pipeline

The OmniHealthFM workflow consists of four major stages:

## Step 1 – Build Gene-Level Text Knowledge

Script: `step1_build_gene_knowledge.py`

This step constructs gene-level textual knowledge by integrating:

- Gene Ontology (GO) annotations
- STRING protein interaction data

Outputs:
- gene_string.jsonl
- gene_string_GO.jsonl
- gene_string_GO_modeling.jsonl

External files required:
- **GO ontology file (go.obo)**
  Download from: http://purl.obolibrary.org/obo/go.obo

- **GOA human annotation file (goa_human.gaf.gz)**  
  Download from: http://current.geneontology.org/products/pages/downloads.html

---

## Step 2 – Llama-3 Fine-Tuning (LoRA)

Script: `step2_finetune_llama.py`

This step fine-tunes a pretrained Llama-3 model using LoRA on gene-level textual knowledge generated in Step 1.

Method:
- 4-bit quantization (QLoRA-style)
- LoRA adapters applied to attention layers

Output:
- LoRA adapter checkpoints

Note:
Access permission to Meta Llama-3 on Hugging Face is required.

---

## Step 3 – Extract Gene Embeddings

Script: `step3_extract_llama_embeddings.py`

This step extracts gene embeddings from the fine-tuned Llama model using mean pooling over last-layer hidden states.

Input:
- `gene_list.csv`: a list of gene symbols used to construct gene-level textual prompts

Output:
- `Gene_Embedding.pkl`: a gene embedding matrix (genes × embedding dimension)

This embedding matrix serves as the fixed gene-level prior in OmniHealthFM.

---

## Step 4 – OmniHealthFM Pretraining

Notebook: `step4_pretrain_OmniHealthFM.ipynb`

Core model implementation:
- OmniHealthFM_Model (see OmniHealthFM.py)

Model configuration:
- config.py

Architecture components include:
- Transformer-based encoder
- Cross-attention with gene embeddings
- Transformer decoder for reconstruction

The model reconstructs masked gene expression values while integrating prior biological knowledge encoded in gene embeddings derived from the fine-tuned Llama model.


# Contact information

Please contact our team if you have any questions:

Jingwen Yan(Jingwen.Yan@UTSouthwestern.edu)

Lei Yu1 (Lei.Yu@UTSouthwestern.edu)

Xue Xiao (Xiao.Xue@UTSouthwestern.edu)

Lin Xu (Lin.Xu@UTSouthwestern.edu)


# Copyright information 

Please see the "LICENSE" file for the copyright information.
