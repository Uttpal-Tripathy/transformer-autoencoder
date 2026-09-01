from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 1536
    n_heads: int = 24
    n_encoder_layers: int = 15
    n_decoder_layers: int = 15
    d_ff: int = 6144
    max_seq_len: int = 1024
    dropout: float = 0.1
    rms_norm_eps: float = 1e-6
    tie_embeddings: bool = True

    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3
    mask_id: int = 4

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
