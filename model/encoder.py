import torch
import torch.nn as nn

from .config import ModelConfig
from .layers import RMSNorm, SelfAttention, FeedForward


class EncoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.attn = SelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), causal=False, key_padding_mask=padding_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.n_encoder_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, padding_mask)
        return self.final_norm(x)
