"""End-to-end demo run: train a tokenizer on data/sample_corpus.txt, train the real
TransformerAutoencoder at a GPU-appropriate size, and show a reconstruction example.

This is a demonstration run (small corpus, small model), not a pretraining run --
see README.md for how to scale up with a real corpus and the full ~1B config.
"""
import argparse
import random
import sys
from pathlib import Path

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer

from model.config import ModelConfig
from model.autoencoder import TransformerAutoencoder
from noise import corrupt
from tokenizer.train_tokenizer import train as train_tokenizer, SPECIAL_TOKENS


def load_lines(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip()]


def encode_batch(tok, cfg, texts, seq_len):
    batch_ids = []
    for text in texts:
        ids = [cfg.bos_id] + tok.encode(text).ids[: seq_len - 2] + [cfg.eos_id]
        ids = ids + [cfg.pad_id] * (seq_len - len(ids))
        batch_ids.append(ids[:seq_len])
    return torch.tensor(batch_ids, dtype=torch.long)


def shift_right(ids, bos_id):
    shifted = ids.new_full(ids.shape, bos_id)
    shifted[:, 1:] = ids[:, :-1]
    return shifted


def decode_ids(tok, ids):
    return tok.decode([i for i in ids if i >= 5])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data/sample_corpus.txt")
    p.add_argument("--tokenizer-dir", default="tokenizer/vocab")
    p.add_argument("--vocab-size", type=int, default=4000)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--checkpoint", default="checkpoints/demo_model.pt")
    p.add_argument("--loss-plot", default="docs/demo_loss_curve.png")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = args.device
    print(f"device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\n=== training tokenizer ===")
    train_tokenizer(args.corpus, args.tokenizer_dir, args.vocab_size)
    tok = ByteLevelBPETokenizer(f"{args.tokenizer_dir}/vocab.json", f"{args.tokenizer_dir}/merges.txt")
    actual_vocab_size = tok.get_vocab_size()
    print(f"actual trained vocab size: {actual_vocab_size}")

    cfg = ModelConfig(
        vocab_size=actual_vocab_size,
        d_model=512, n_heads=8, n_encoder_layers=6, n_decoder_layers=6,
        d_ff=2048, max_seq_len=args.seq_len,
    )

    lines = load_lines(args.corpus)
    print(f"\n=== corpus: {len(lines)} lines ===")

    print("\n=== building model ===")
    model = TransformerAutoencoder(cfg).to(device)
    print(f"parameters: {model.num_parameters():,}  (~{model.num_parameters()/1e6:.1f}M)")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = random.Random(0)

    print(f"\n=== training for {args.steps} steps ===")
    model.train()
    losses = []
    for step in range(1, args.steps + 1):
        texts = [rng.choice(lines) for _ in range(args.batch_size)]
        target_ids = encode_batch(tok, cfg, texts, args.seq_len).to(device)

        noisy_ids = target_ids.clone()
        for i in range(noisy_ids.size(0)):
            row = [t for t in target_ids[i].tolist() if t != cfg.pad_id]
            noised = corrupt(row, cfg, rng=rng)[: args.seq_len]
            noised = noised + [cfg.pad_id] * (args.seq_len - len(noised))
            noisy_ids[i] = torch.tensor(noised, dtype=torch.long, device=device)

        decoder_input_ids = shift_right(target_ids, cfg.bos_id)
        encoder_padding_mask = noisy_ids != cfg.pad_id

        logits = model(noisy_ids, decoder_input_ids, encoder_padding_mask)
        loss = F.cross_entropy(
            logits.view(-1, cfg.vocab_size), target_ids.view(-1), ignore_index=cfg.pad_id
        )

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

        if step % 20 == 0 or step == 1:
            print(f"step {step:4d}/{args.steps}  loss {loss.item():.4f}")

    if device == "cuda":
        print(f"\npeak VRAM used: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": cfg}, ckpt_path)
    print(f"saved checkpoint: {ckpt_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4.5))
        plt.plot(losses)
        plt.xlabel("step")
        plt.ylabel("cross-entropy loss")
        plt.title(f"Transformer autoencoder demo training ({model.num_parameters()/1e6:.1f}M params, {device})")
        plt.tight_layout()
        plot_path = Path(args.loss_plot)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=120)
        print(f"saved loss curve: {plot_path}")
    except ImportError:
        print("matplotlib not installed, skipping loss curve plot")

    print("\n=== reconstruction demo ===")
    model.eval()
    for text in [lines[0], lines[len(lines) // 2], lines[-1]]:
        target_ids = encode_batch(tok, cfg, [text], args.seq_len).to(device)
        row = [t for t in target_ids[0].tolist() if t != cfg.pad_id]
        noised = corrupt(row, cfg, rng=rng)[: args.seq_len]
        noised = noised + [cfg.pad_id] * (args.seq_len - len(noised))
        noisy_ids = torch.tensor([noised], dtype=torch.long, device=device)
        encoder_padding_mask = noisy_ids != cfg.pad_id

        with torch.no_grad():
            memory = model.encoder(model.embed(noisy_ids), encoder_padding_mask)
            generated = torch.full((1, 1), cfg.bos_id, dtype=torch.long, device=device)
            recent = []
            for _ in range(args.seq_len - 1):
                hidden = model.decoder(model.embed(generated), memory, encoder_padding_mask)
                next_logits = model.project_to_vocab(hidden[:, -1:]).clone()
                for tok_id in set(recent[-4:]):
                    next_logits[0, 0, tok_id] -= 8.0  # discourage immediate repetition loops
                next_token = next_logits.argmax(dim=-1)
                recent.append(next_token.item())
                generated = torch.cat([generated, next_token], dim=1)
                if next_token.item() == cfg.eos_id:
                    break

        print(f"\noriginal:      {text}")
        print(f"corrupted:     {decode_ids(tok, noised)}")
        print(f"reconstructed: {decode_ids(tok, generated[0].tolist())}")


if __name__ == "__main__":
    main()
