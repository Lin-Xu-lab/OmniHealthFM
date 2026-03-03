"""
step4_OmniHealthFM_model.py

What this file contains
-----------------------
Implementation of the OmniHealthFM foundation model for medical prediction tasks.

OmniHealthFM is composed of two main components:
(1) a gene embedding module providing fixed gene-level representations, and
(2) a masked reconstruction module operating on partially observed gene
    expression profiles.

The overall architecture and pretraining workflow correspond to those
illustrated in Figure 1 of the manuscript.

Model overview
--------------
During pretraining, bulk gene expression profiles are randomly masked and
processed by a Transformer-based encoder. The encoder produces token-level
representations together with a designated global representation (CLS) token
that summarizes sample-level information.

Fixed gene embeddings are incorporated through a cross-attention mechanism,
in which the global representation token attends to the full set of gene
embeddings, yielding an integrated sample-level representation that combines
expression-derived features with external gene-level knowledge.

A Transformer-based decoder then reconstructs masked gene expression values
using the integrated global representation together with encoder-derived
features for observed genes.

Important note on gene embeddings
---------------------------------
The input `gene_embedding` to this model is a fixed gene-level representation
matrix generated in Step 3
(`step3_extract_llama_embeddings.py`) from a fine-tuned LLaMA-3 language model
and stored as:

    models/Gene_Embedding.pkl

These gene embeddings constitute the gene embedding module of OmniHealthFM.
They are treated as frozen external knowledge priors and are not updated
during OmniHealthFM pretraining or downstream training.

Main components
---------------
- PatchShuffle:
  Random masking and shuffling of gene tokens (MAE-style masking).
- RNA_Encoder:
  Transformer-based encoder operating on partially observed gene expression,
  producing token-level features and a global (CLS) representation.
- CrossAttention:
  Cross-attention mechanism in which the global representation attends to the
  full gene embedding table.
- RNA_Decoder:
  Transformer-based decoder that reconstructs masked gene expression values.
- OmniHealthFMModel:
  Full OmniHealthFM model integrating all components.

Dependencies
------------
torch, numpy, einops, timm
"""

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block


# ----------------------- utils -----------------------
def random_indexes(size: int):
    """Return a random permutation and its inverse (forward, backward)."""
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes


def take_indexes(sequences: torch.Tensor, indexes: torch.Tensor):
    """
    Gather along dim=0 with broadcasted indexes.
    sequences: (T, B, C)
    indexes:   (T, B)
    returns:   (T, B, C)
    """
    return torch.gather(
        sequences,
        0,
        repeat(indexes, 't b -> t b c', c=sequences.shape[-1])
    )


class PatchShuffle(nn.Module):
    """Shuffle tokens per sample and keep only a (1 - ratio) subset."""
    def __init__(self, ratio: float):
        super().__init__()
        self.ratio = ratio

    def forward(self, patches: torch.Tensor):
        """
        patches: (T, B, C)
        returns:
          patches_kept: (T_kept, B, C)
          fwd_idx:      (T, B)  permutation used to shuffle
          bwd_idx:      (T, B)  inverse permutation to unshuffle
        """
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        idx_pairs = [random_indexes(T) for _ in range(B)]
        fwd_idx = torch.as_tensor(
            np.stack([p[0] for p in idx_pairs], axis=-1),
            dtype=torch.long,
            device=patches.device
        )
        bwd_idx = torch.as_tensor(
            np.stack([p[1] for p in idx_pairs], axis=-1),
            dtype=torch.long,
            device=patches.device
        )

        patches = take_indexes(patches, fwd_idx)
        patches = patches[:remain_T]
        return patches, fwd_idx, bwd_idx


# ----------------------- encoder/decoder with gene-ID embedding -----------------------
class RNA_Encoder(nn.Module):
    """
    Encoder that adds per-gene identity embedding BEFORE shuffle/cropping.
    """
    def __init__(self, emb_dim=128, RNA_tokens=2000,
                 encoder_head=4, encoder_layer=6, mask_ratio=0.1):
        super().__init__()
        self.embedding = nn.Linear(1, emb_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)
        self.transformer = nn.Sequential(
            *[Block(emb_dim, encoder_head, attn_drop=0.1, proj_drop=0.1)
              for _ in range(encoder_layer)]
        )
        trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor, gene_id_embed: torch.Tensor):
        """
        x:             (B, T) normalized expression
        gene_id_embed: (T, C) per-gene identity embeddings
        """
        x = self.embedding(x.unsqueeze(-1)) + gene_id_embed.unsqueeze(0)
        x = rearrange(x, 'b t c -> t b c')
        x, _, bwd_idx = self.shuffle(x)

        cls = self.cls_token.expand(-1, x.size(1), -1)
        x = torch.cat([cls, x], dim=0)

        x = rearrange(x, 't b c -> b t c')
        x = self.transformer(x)
        x = rearrange(x, 'b t c -> t b c')

        return x, bwd_idx


class RNA_Decoder(nn.Module):
    """
    Decoder that adds per-gene identity embedding AFTER unshuffle/restoration.
    """
    def __init__(self, emb_dim=128, decoder_head=4, decoder_layer=2):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.transformer = nn.Sequential(
            *[Block(emb_dim, decoder_head, attn_drop=0.1, proj_drop=0.1)
              for _ in range(decoder_layer)]
        )
        self.decoding = nn.Linear(emb_dim, 1)
        trunc_normal_(self.mask_token, std=0.02)

    def forward(self, features, backward_indexes, gene_id_embed):
        T_vis_plus_cls, B, C = features.shape
        T_all_plus_cls = backward_indexes.shape[0] + 1
        n_masked = T_all_plus_cls - T_vis_plus_cls
        T_all = T_all_plus_cls - 1

        features = torch.cat(
            [features, self.mask_token.expand(n_masked, B, C)],
            dim=0
        )

        full_idx = torch.cat(
            [torch.zeros(1, B, dtype=torch.long, device=features.device),
             backward_indexes + 1],
            dim=0
        )
        features = take_indexes(features, full_idx)

        gene_id_embed_b = gene_id_embed.unsqueeze(1).expand(-1, B, -1)
        features = torch.cat([features[:1], features[1:] + gene_id_embed_b], dim=0)

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')

        gene_preds = self.decoding(features[1:]).squeeze(-1).permute(1, 0)
        mask = torch.zeros_like(gene_preds)

        for b in range(B):
            masked_idx = full_idx[-n_masked:, b] - 1
            mask[b, masked_idx] = 1

        return gene_preds, mask, [features[0]]


# ----------------------- attention -----------------------
class CrossAttention(nn.Module):
    """Single-head cross-attention."""
    def __init__(self, emb_dim):
        super().__init__()
        self.q = nn.Linear(emb_dim, emb_dim)
        self.k = nn.Linear(emb_dim, emb_dim)
        self.v = nn.Linear(emb_dim, emb_dim)
        self.scale = emb_dim ** -0.5

    def forward(self, query, key, value):
        attn = (self.q(query) @ self.k(key).transpose(-2, -1)) * self.scale
        w = torch.softmax(attn, dim=-1)
        return w @ self.v(value)


# ----------------------- full model -----------------------
class OmniHealthFM_Model(nn.Module):
    """
    OmniHealthFM core model.

    gene_embedding: fixed gene-level embedding matrix loaded from
                    models/Gene_Embedding.pkl (Step 3)
    """
    def __init__(self, config, gene_embedding: torch.Tensor):
        super().__init__()
        self.config = config

        self.gene_embedding = gene_embedding.to(config.device).detach()
        self.gene_embedding.requires_grad_(False)

        self.embedding_proj = nn.Linear(4096, config.emb_dim)

        self.encoder = RNA_Encoder(
            emb_dim=config.emb_dim,
            RNA_tokens=config.RNA_tokens,
            encoder_head=config.encoder_head,
            encoder_layer=config.encoder_layer,
            mask_ratio=config.mask_ratio
        )
        self.decoder = RNA_Decoder(
            emb_dim=config.emb_dim,
            decoder_head=config.decoder_head,
            decoder_layer=config.decoder_layer
        )

        self.cross_attn = CrossAttention(config.emb_dim)

    def forward(self, patches1: torch.Tensor):
        B, T = patches1.shape
        gene_id_embed = self.embedding_proj(self.gene_embedding)

        enc_feats, bwd_idx = self.encoder(patches1, gene_id_embed)

        feats_btc = rearrange(enc_feats, 't b c -> b t c')
        cls = feats_btc[:, :1]
        E = gene_id_embed.unsqueeze(0).expand(B, -1, -1)

        cls_updated = self.cross_attn(cls, E, E)
        feats_btc = torch.cat([cls_updated, feats_btc[:, 1:]], dim=1)

        feats_tbc = rearrange(feats_btc, 'b t c -> t b c') + enc_feats
        recon, mask, decode_cls = self.decoder(feats_tbc, bwd_idx, gene_id_embed)

        return recon, mask, cls_updated, feats_tbc, decode_cls