from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class DiffusionLMConfig:
    vocab_size: int
    seq_len: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    dropout: float
    diffusion_steps: int


class DiffusionTransformerLM(nn.Module):
    def __init__(self, cfg: DiffusionLMConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.time_emb = nn.Embedding(cfg.diffusion_steps + 1, cfg.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, input_ids, timesteps, attention_mask=None):
        B, L = input_ids.shape
        if L > self.cfg.seq_len:
            raise ValueError(f"Sequence length {L} > cfg.seq_len {self.cfg.seq_len}")

        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)

        t_emb = self.time_emb(timesteps).unsqueeze(1)
        x = self.drop(x + t_emb)

        src_key_padding_mask = None if attention_mask is None else ~attention_mask.bool()
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.ln_f(x)
        return self.lm_head(x)
