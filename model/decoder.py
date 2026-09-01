import torch
import torch.nn as nn

from .config import ModelConfig
from .layers import RMSNorm, SelfAttention, CrossAttention, FeedForward


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.self_attn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.self_attn = SelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.cross_attn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.cross_attn = CrossAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                memory_padding_mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.self_attn(self.self_attn_norm(x), causal=True)
        x = x + self.cross_attn(self.cross_attn_norm(x), memory, memory_padding_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_decoder_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                memory_padding_mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, memory_padding_mask)
        return self.final_norm(x)
