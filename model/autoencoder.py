import torch
import torch.nn as nn

from .config import ModelConfig
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder


class TransformerAutoencoder(nn.Module):
    """Denoising transformer autoencoder (BART/T5-style encoder-decoder).

    Pipeline: tokenize -> corrupt (mask/delete/replace, see noise.py) -> encode
    corrupted tokens into latent Z -> decode with cross-attention into Z,
    teacher-forced on the *original* tokens -> project to vocab -> cross-entropy
    against the original (uncorrupted) sequence.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.encoder = TransformerEncoder(cfg)
        self.decoder = TransformerDecoder(cfg)
        self.lm_head = None if cfg.tie_embeddings else nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        t = ids.shape[1]
        positions = torch.arange(t, device=ids.device).unsqueeze(0)
        return self.token_embed(ids) + self.pos_embed(positions)

    def project_to_vocab(self, hidden: torch.Tensor) -> torch.Tensor:
        weight = self.token_embed.weight if self.lm_head is None else self.lm_head.weight
        return hidden @ weight.T

    def forward(self, noisy_ids: torch.Tensor, decoder_input_ids: torch.Tensor,
                encoder_padding_mask: torch.Tensor = None) -> torch.Tensor:
        z = self.encoder(self.embed(noisy_ids), encoder_padding_mask)
        hidden = self.decoder(self.embed(decoder_input_ids), z, encoder_padding_mask)
        return self.project_to_vocab(hidden)

    def num_parameters(self, trainable_only: bool = False) -> int:
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)
