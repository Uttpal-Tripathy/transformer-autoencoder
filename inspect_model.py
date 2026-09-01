import torch

from model.config import ModelConfig
from model.autoencoder import TransformerAutoencoder


def main():
    cfg = ModelConfig()
    model = TransformerAutoencoder(cfg)
    n = model.num_parameters()
    print(f"config: d_model={cfg.d_model} n_heads={cfg.n_heads} "
          f"enc_layers={cfg.n_encoder_layers} dec_layers={cfg.n_decoder_layers} "
          f"d_ff={cfg.d_ff} vocab={cfg.vocab_size} tied_embeddings={cfg.tie_embeddings}")
    print(f"total parameters: {n:,}  (~{n / 1e9:.3f}B)")

    b, t = 2, 32
    noisy = torch.randint(5, cfg.vocab_size, (b, t))
    dec_in = torch.randint(5, cfg.vocab_size, (b, t))
    with torch.no_grad():
        logits = model(noisy, dec_in)
    print(f"forward pass ok, logits shape: {tuple(logits.shape)}")
    assert logits.shape == (b, t, cfg.vocab_size)


if __name__ == "__main__":
    main()
