"""Demo-scale training loop for the transformer autoencoder.

This proves the architecture, noising, and loss wiring are correct end to end.
It is NOT a real 1B-parameter pretraining run: reaching convergence at this
scale needs a corpus of many billions of tokens and days-to-weeks of
multi-GPU compute, neither of which this script assumes. Point it at a
larger corpus and a GPU box to scale up.
"""
import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer

from model.config import ModelConfig
from model.autoencoder import TransformerAutoencoder
from noise import corrupt


def load_lines(path: str, max_lines: int = None):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    lines = [l.strip() for l in lines if l.strip()]
    return lines[:max_lines] if max_lines else lines


def encode_batch(tok: ByteLevelBPETokenizer, cfg: ModelConfig, texts, seq_len: int):
    batch_ids = []
    for text in texts:
        ids = [cfg.bos_id] + tok.encode(text).ids[: seq_len - 2] + [cfg.eos_id]
        ids = ids + [cfg.pad_id] * (seq_len - len(ids))
        batch_ids.append(ids[:seq_len])
    return torch.tensor(batch_ids, dtype=torch.long)


def shift_right(ids: torch.Tensor, bos_id: int) -> torch.Tensor:
    shifted = ids.new_full(ids.shape, bos_id)
    shifted[:, 1:] = ids[:, :-1]
    return shifted


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data/corpus.txt")
    p.add_argument("--tokenizer-dir", default="tokenizer/vocab")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--checkpoint", default="checkpoints/model.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = ModelConfig(max_seq_len=args.seq_len)
    tok = ByteLevelBPETokenizer(
        f"{args.tokenizer_dir}/vocab.json", f"{args.tokenizer_dir}/merges.txt"
    )

    lines = load_lines(args.corpus)
    if not lines:
        raise SystemExit(f"no text found in {args.corpus} -- run data/generate_corpus.py first")

    model = TransformerAutoencoder(cfg).to(args.device)
    print(f"model parameters: {model.num_parameters():,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = random.Random(0)

    model.train()
    for step in range(1, args.steps + 1):
        texts = [rng.choice(lines) for _ in range(args.batch_size)]
        target_ids = encode_batch(tok, cfg, texts, args.seq_len).to(args.device)

        noisy_ids = target_ids.clone()
        for i in range(noisy_ids.size(0)):
            row = [t for t in target_ids[i].tolist() if t != cfg.pad_id]
            noised = corrupt(row, cfg, rng=rng)[: args.seq_len]
            noised = noised + [cfg.pad_id] * (args.seq_len - len(noised))
            noisy_ids[i] = torch.tensor(noised, dtype=torch.long, device=args.device)

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

        if step % 5 == 0 or step == 1:
            print(f"step {step}/{args.steps}  loss {loss.item():.4f}")

    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": cfg}, ckpt_path)
    print(f"saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
