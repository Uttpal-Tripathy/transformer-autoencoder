"""End-to-end demo of the RAE-style downstream capabilities built on top of the same
latent Z as the base autoencoder:

    Encoder -> latent Z -> Reconstruction | Contrastive | Classification | Retrieval | Anomaly

Trains the base autoencoder on data/sample_corpus.tsv (labeled with topic categories),
then exercises each downstream head against the trained encoder. Demonstration scale,
same honesty caveats as run_demo.py: small corpus, not a pretraining run.
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
from model.downstream import pool_latent, contrastive_loss, ClassificationHead, nearest_neighbors, reconstruction_error
from noise import corrupt
from tokenizer.train_tokenizer import train as train_tokenizer
from run_demo import encode_batch, shift_right, decode_ids


def load_labeled_corpus(path):
    labels, texts = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        label, text = line.split("\t", 1)
        labels.append(label)
        texts.append(text)
    return labels, texts


def encode_to_latent(model, tok, cfg, texts, seq_len, device):
    ids = encode_batch(tok, cfg, texts, seq_len).to(device)
    mask = ids != cfg.pad_id
    with torch.no_grad():
        z = model.encoder(model.embed(ids), mask)
    return pool_latent(z, mask), ids, mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data/sample_corpus.tsv")
    p.add_argument("--tokenizer-dir", default="tokenizer/vocab")
    p.add_argument("--vocab-size", type=int, default=4000)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--recon-steps", type=int, default=1500)
    p.add_argument("--classifier-steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = args.device
    print(f"device: {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    labels, texts = load_labeled_corpus(args.corpus)
    categories = sorted(set(labels))
    label_to_idx = {c: i for i, c in enumerate(categories)}
    print(f"\n=== corpus: {len(texts)} lines, {len(categories)} categories: {categories} ===")

    plain_corpus_path = Path("data/sample_corpus.txt")
    plain_corpus_path.write_text("\n".join(texts) + "\n", encoding="utf-8")

    print("\n=== training tokenizer ===")
    train_tokenizer(str(plain_corpus_path), args.tokenizer_dir, args.vocab_size)
    tok = ByteLevelBPETokenizer(f"{args.tokenizer_dir}/vocab.json", f"{args.tokenizer_dir}/merges.txt")
    actual_vocab_size = tok.get_vocab_size()
    print(f"actual trained vocab size: {actual_vocab_size}")

    cfg = ModelConfig(
        vocab_size=actual_vocab_size,
        d_model=512, n_heads=8, n_encoder_layers=6, n_decoder_layers=6,
        d_ff=2048, max_seq_len=args.seq_len,
    )
    model = TransformerAutoencoder(cfg).to(device)
    print(f"\n=== base autoencoder: {model.num_parameters():,} params ===")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = random.Random(0)

    # ---- Phase 1: Reconstruction (the base objective the latent Z is built on) ----
    print(f"\n=== phase 1/4: reconstruction training ({args.recon_steps} steps) ===")
    model.train()
    for step in range(1, args.recon_steps + 1):
        batch_texts = [rng.choice(texts) for _ in range(args.batch_size)]
        target_ids = encode_batch(tok, cfg, batch_texts, args.seq_len).to(device)

        noisy_ids = target_ids.clone()
        for i in range(noisy_ids.size(0)):
            row = [t for t in target_ids[i].tolist() if t != cfg.pad_id]
            noised = corrupt(row, cfg, rng=rng)[: args.seq_len]
            noised = noised + [cfg.pad_id] * (args.seq_len - len(noised))
            noisy_ids[i] = torch.tensor(noised, dtype=torch.long, device=device)

        decoder_input_ids = shift_right(target_ids, cfg.bos_id)
        encoder_padding_mask = noisy_ids != cfg.pad_id

        logits = model(noisy_ids, decoder_input_ids, encoder_padding_mask)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), target_ids.view(-1), ignore_index=cfg.pad_id)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 300 == 0 or step == 1:
            print(f"  step {step:4d}/{args.recon_steps}  loss {loss.item():.4f}")

    # ---- Phase 2: Contrastive -- two independently-corrupted views should land close in Z ----
    print("\n=== phase 2/4: contrastive check (two noise views of the same sentences) ===")
    model.eval()
    sample_texts = [rng.choice(texts) for _ in range(8)]
    ids = encode_batch(tok, cfg, sample_texts, args.seq_len).to(device)

    def noisy_view():
        v = ids.clone()
        for i in range(v.size(0)):
            row = [t for t in ids[i].tolist() if t != cfg.pad_id]
            noised = corrupt(row, cfg, rng=rng)[: args.seq_len]
            noised = noised + [cfg.pad_id] * (args.seq_len - len(noised))
            v[i] = torch.tensor(noised, dtype=torch.long, device=device)
        return v

    view_a, view_b = noisy_view(), noisy_view()
    with torch.no_grad():
        z_a = pool_latent(model.encoder(model.embed(view_a), view_a != cfg.pad_id), view_a != cfg.pad_id)
        z_b = pool_latent(model.encoder(model.embed(view_b), view_b != cfg.pad_id), view_b != cfg.pad_id)
    same_pair_loss = contrastive_loss(z_a, z_b).item()
    shuffled_loss = contrastive_loss(z_a, z_b[torch.randperm(z_b.size(0))]).item()
    print(f"  contrastive loss, same underlying sentence (2 noise views): {same_pair_loss:.4f}")
    print(f"  contrastive loss, mismatched/shuffled pairs:                {shuffled_loss:.4f}")
    print("  (lower = more similar; same-sentence pairs should score lower than shuffled ones)")

    # ---- Phase 3: Classification head on pooled Z ----
    print(f"\n=== phase 3/4: classification head ({args.classifier_steps} steps, {len(categories)}-way) ===")
    clf = ClassificationHead(cfg.d_model, len(categories)).to(device)
    clf_opt = torch.optim.AdamW(clf.parameters(), lr=1e-3)
    label_ids = torch.tensor([label_to_idx[l] for l in labels], dtype=torch.long)

    for step in range(1, args.classifier_steps + 1):
        idx = [rng.randrange(len(texts)) for _ in range(args.batch_size)]
        batch_texts = [texts[i] for i in idx]
        batch_labels = label_ids[idx].to(device)

        pooled, _, _ = encode_to_latent(model, tok, cfg, batch_texts, args.seq_len, device)
        clf_logits = clf(pooled)
        clf_loss = F.cross_entropy(clf_logits, batch_labels)

        clf_opt.zero_grad()
        clf_loss.backward()
        clf_opt.step()
        if step % 100 == 0 or step == 1:
            acc = (clf_logits.argmax(-1) == batch_labels).float().mean().item()
            print(f"  step {step:4d}/{args.classifier_steps}  loss {clf_loss.item():.4f}  batch acc {acc:.2f}")

    with torch.no_grad():
        all_pooled, _, _ = encode_to_latent(model, tok, cfg, texts, args.seq_len, device)
        all_logits = clf(all_pooled)
        full_acc = (all_logits.argmax(-1) == label_ids.to(device)).float().mean().item()
    print(f"  final accuracy over the full (training) corpus: {full_acc:.2%}")

    # ---- Phase 4: Retrieval + anomaly detection ----
    print("\n=== phase 4/4: retrieval + anomaly detection ===")
    corpus_pooled, _, _ = encode_to_latent(model, tok, cfg, texts, args.seq_len, device)

    query = texts[0]
    query_pooled, _, _ = encode_to_latent(model, tok, cfg, [query], args.seq_len, device)
    top_idx, top_scores = nearest_neighbors(query_pooled[0], corpus_pooled, k=4)
    print(f"\n  retrieval query: {query}  [{labels[0]}]")
    for i, score in zip(top_idx.tolist(), top_scores.tolist()):
        marker = " (self)" if texts[i] == query else ""
        print(f"    {score:.3f}  [{labels[i]}] {texts[i]}{marker}")

    normal_text = texts[len(texts) // 2]
    weird_text = "purple elephants compute quarterly firewalls beneath the singing algorithm ocean"
    for label, text in [("in-distribution", normal_text), ("out-of-distribution", weird_text)]:
        target_ids = encode_batch(tok, cfg, [text], args.seq_len).to(device)
        row = [t for t in target_ids[0].tolist() if t != cfg.pad_id]
        noised = corrupt(row, cfg, rng=rng)[: args.seq_len]
        noised = noised + [cfg.pad_id] * (args.seq_len - len(noised))
        noisy_ids = torch.tensor([noised], dtype=torch.long, device=device)
        decoder_input_ids = shift_right(target_ids, cfg.bos_id)
        score = reconstruction_error(model, target_ids, noisy_ids, decoder_input_ids, cfg.pad_id, noisy_ids != cfg.pad_id)
        print(f"\n  anomaly score ({label}): {score.item():.3f}   \"{text}\"")

    print("\n=== done: reconstruction, contrastive, classification, retrieval, and anomaly detection all exercised on the same latent Z ===")


if __name__ == "__main__":
    main()
