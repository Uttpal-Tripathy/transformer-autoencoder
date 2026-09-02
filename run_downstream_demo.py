"""End-to-end demo of the RAE-style downstream capabilities built on top of the same
latent Z as the base autoencoder:

    Encoder -> latent Z -> Reconstruction | Contrastive | Classification | Retrieval | Anomaly

Trains the base autoencoder on data/sample_corpus.tsv (labeled with topic categories),
exercises each downstream head against the trained encoder, and saves a graph for each
head to docs/. Demonstration scale, same honesty caveats as run_demo.py: small corpus,
not a pretraining run.

The phase_* functions are reusable (imported by notebooks/05_downstream_demo.ipynb) so
the script and the notebook share one implementation instead of drifting apart.
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
from run_demo import encode_batch, shift_right


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


def build_model_and_tokenizer(corpus_path, tokenizer_dir, vocab_size, seq_len, device):
    train_tokenizer(corpus_path, tokenizer_dir, vocab_size)
    tok = ByteLevelBPETokenizer(f"{tokenizer_dir}/vocab.json", f"{tokenizer_dir}/merges.txt")
    cfg = ModelConfig(
        vocab_size=tok.get_vocab_size(),
        d_model=512, n_heads=8, n_encoder_layers=6, n_decoder_layers=6,
        d_ff=2048, max_seq_len=seq_len,
    )
    model = TransformerAutoencoder(cfg).to(device)
    return tok, cfg, model


def encode_to_latent(model, tok, cfg, texts, seq_len, device):
    ids = encode_batch(tok, cfg, texts, seq_len).to(device)
    mask = ids != cfg.pad_id
    with torch.no_grad():
        z = model.encoder(model.embed(ids), mask)
    return pool_latent(z, mask), ids, mask


def _noisy_ids(ids, cfg, seq_len, device, rng):
    out = ids.clone()
    for i in range(out.size(0)):
        row = [t for t in ids[i].tolist() if t != cfg.pad_id]
        noised = corrupt(row, cfg, rng=rng)[:seq_len]
        noised = noised + [cfg.pad_id] * (seq_len - len(noised))
        out[i] = torch.tensor(noised, dtype=torch.long, device=device)
    return out


def phase_reconstruction(model, tok, cfg, texts, opt, steps, batch_size, seq_len, device, rng, log_every=300):
    """Trains the base denoising objective. Returns the per-step loss history."""
    model.train()
    losses = []
    for step in range(1, steps + 1):
        batch_texts = [rng.choice(texts) for _ in range(batch_size)]
        target_ids = encode_batch(tok, cfg, batch_texts, seq_len).to(device)
        noisy_ids = _noisy_ids(target_ids, cfg, seq_len, device, rng)
        decoder_input_ids = shift_right(target_ids, cfg.bos_id)
        encoder_padding_mask = noisy_ids != cfg.pad_id

        logits = model(noisy_ids, decoder_input_ids, encoder_padding_mask)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), target_ids.view(-1), ignore_index=cfg.pad_id)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if log_every and (step % log_every == 0 or step == 1):
            print(f"  step {step:4d}/{steps}  loss {loss.item():.4f}")
    return losses


def phase_contrastive(model, tok, cfg, texts, seq_len, device, rng, n=8):
    """Two independently-corrupted views of the same sentence should land close in Z."""
    model.eval()
    sample_texts = [rng.choice(texts) for _ in range(n)]
    ids = encode_batch(tok, cfg, sample_texts, seq_len).to(device)

    view_a = _noisy_ids(ids, cfg, seq_len, device, rng)
    view_b = _noisy_ids(ids, cfg, seq_len, device, rng)
    with torch.no_grad():
        z_a = pool_latent(model.encoder(model.embed(view_a), view_a != cfg.pad_id), view_a != cfg.pad_id)
        z_b = pool_latent(model.encoder(model.embed(view_b), view_b != cfg.pad_id), view_b != cfg.pad_id)
    same_pair_loss = contrastive_loss(z_a, z_b).item()
    shuffled_loss = contrastive_loss(z_a, z_b[torch.randperm(z_b.size(0))]).item()
    return same_pair_loss, shuffled_loss


def phase_classification(model, tok, cfg, texts, labels, categories, opt_lr, steps, batch_size, seq_len, device, rng, log_every=100):
    """Trains a linear head on pooled Z to predict topic category. Returns (clf, losses, accs, full_acc)."""
    label_to_idx = {c: i for i, c in enumerate(categories)}
    label_ids = torch.tensor([label_to_idx[l] for l in labels], dtype=torch.long)

    clf = ClassificationHead(cfg.d_model, len(categories)).to(device)
    clf_opt = torch.optim.AdamW(clf.parameters(), lr=opt_lr)

    losses, accs = [], []
    for step in range(1, steps + 1):
        idx = [rng.randrange(len(texts)) for _ in range(batch_size)]
        batch_texts = [texts[i] for i in idx]
        batch_labels = label_ids[idx].to(device)

        pooled, _, _ = encode_to_latent(model, tok, cfg, batch_texts, seq_len, device)
        clf_logits = clf(pooled)
        clf_loss = F.cross_entropy(clf_logits, batch_labels)

        clf_opt.zero_grad()
        clf_loss.backward()
        clf_opt.step()

        acc = (clf_logits.argmax(-1) == batch_labels).float().mean().item()
        losses.append(clf_loss.item())
        accs.append(acc)
        if log_every and (step % log_every == 0 or step == 1):
            print(f"  step {step:4d}/{steps}  loss {clf_loss.item():.4f}  batch acc {acc:.2f}")

    with torch.no_grad():
        all_pooled, _, _ = encode_to_latent(model, tok, cfg, texts, seq_len, device)
        full_acc = (clf(all_pooled).argmax(-1) == label_ids.to(device)).float().mean().item()
    return clf, losses, accs, full_acc


def phase_retrieval(model, tok, cfg, texts, labels, seq_len, device, query_idx=0, k=4):
    corpus_pooled, _, _ = encode_to_latent(model, tok, cfg, texts, seq_len, device)
    query_pooled, _, _ = encode_to_latent(model, tok, cfg, [texts[query_idx]], seq_len, device)
    top_idx, top_scores = nearest_neighbors(query_pooled[0], corpus_pooled, k=k)
    return corpus_pooled, top_idx.tolist(), top_scores.tolist()


def phase_anomaly(model, tok, cfg, texts, seq_len, device, rng):
    normal_text = texts[len(texts) // 2]
    weird_text = "purple elephants compute quarterly firewalls beneath the singing algorithm ocean"
    scores = {}
    for label, text in [("in_distribution", normal_text), ("out_of_distribution", weird_text)]:
        target_ids = encode_batch(tok, cfg, [text], seq_len).to(device)
        noisy_ids = _noisy_ids(target_ids, cfg, seq_len, device, rng)
        decoder_input_ids = shift_right(target_ids, cfg.bos_id)
        score = reconstruction_error(model, target_ids, noisy_ids, decoder_input_ids, cfg.pad_id, noisy_ids != cfg.pad_id)
        scores[label] = (text, score.item())
    return scores


def pca_2d(x: torch.Tensor):
    """Project x (n, d) onto its top-2 principal components -- no sklearn dependency needed."""
    centered = x - x.mean(dim=0, keepdim=True)
    _, _, v = torch.linalg.svd(centered, full_matrices=False)
    return (centered @ v[:2].T).cpu().numpy()


def save_all_plots(docs_dir, recon_losses, contrastive_scores, clf_losses, clf_accs,
                    corpus_pooled, labels, categories, anomaly_scores, model_params, device):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    docs_dir = Path(docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Reconstruction loss curve
    plt.figure(figsize=(8, 4.5))
    plt.plot(recon_losses)
    plt.xlabel("step")
    plt.ylabel("cross-entropy loss")
    plt.title(f"Reconstruction training ({model_params/1e6:.1f}M params, {device})")
    plt.tight_layout()
    plt.savefig(docs_dir / "downstream_reconstruction_loss.png", dpi=120)
    plt.close()

    # 2. Contrastive: same-pair vs shuffled-pair loss
    plt.figure(figsize=(5, 4.5))
    bars = plt.bar(["same sentence\n(2 noise views)", "shuffled\n(mismatched)"],
                    contrastive_scores, color=["#4C72B0", "#C44E52"])
    plt.ylabel("contrastive loss (lower = more similar)")
    plt.title("Contrastive: matching vs. mismatched pairs")
    for b, v in zip(bars, contrastive_scores):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(docs_dir / "downstream_contrastive_comparison.png", dpi=120)
    plt.close()

    # 3. Classification loss + accuracy curve
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(clf_losses, color="#4C72B0", label="loss")
    ax1.set_xlabel("step")
    ax1.set_ylabel("cross-entropy loss", color="#4C72B0")
    ax2 = ax1.twinx()
    ax2.plot(clf_accs, color="#55A868", label="batch accuracy")
    ax2.set_ylabel("batch accuracy", color="#55A868")
    ax2.set_ylim(0, 1)
    plt.title(f"Classification head training ({len(categories)}-way topic classification)")
    fig.tight_layout()
    plt.savefig(docs_dir / "downstream_classification_curve.png", dpi=120)
    plt.close()

    # 4. Embedding space (retrieval): 2D PCA of pooled Z, colored by category
    coords = pca_2d(corpus_pooled)
    plt.figure(figsize=(6.5, 6))
    palette = plt.get_cmap("tab10")
    for i, cat in enumerate(categories):
        idx = [j for j, l in enumerate(labels) if l == cat]
        plt.scatter(coords[idx, 0], coords[idx, 1], label=cat, color=palette(i), s=35, alpha=0.8)
    plt.legend(fontsize=8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Latent Z embedding space (PCA), colored by topic")
    plt.tight_layout()
    plt.savefig(docs_dir / "downstream_embedding_space.png", dpi=120)
    plt.close()

    # 5. Anomaly detection: in-distribution vs out-of-distribution reconstruction error
    plt.figure(figsize=(5, 4.5))
    keys = list(anomaly_scores.keys())
    values = [anomaly_scores[k][1] for k in keys]
    bars = plt.bar(["in-distribution", "out-of-distribution"], values, color=["#4C72B0", "#C44E52"])
    plt.ylabel("reconstruction error (anomaly score)")
    plt.title("Anomaly detection via reconstruction error")
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(docs_dir / "downstream_anomaly_scores.png", dpi=120)
    plt.close()

    print(f"saved 5 graphs to {docs_dir}/")


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
    p.add_argument("--docs-dir", default="docs")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = args.device
    print(f"device: {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    labels, texts = load_labeled_corpus(args.corpus)
    categories = sorted(set(labels))
    print(f"\n=== corpus: {len(texts)} lines, {len(categories)} categories: {categories} ===")

    plain_corpus_path = Path("data/sample_corpus.txt")
    plain_corpus_path.write_text("\n".join(texts) + "\n", encoding="utf-8")

    print("\n=== training tokenizer ===")
    tok, cfg, model = build_model_and_tokenizer(
        str(plain_corpus_path), args.tokenizer_dir, args.vocab_size, args.seq_len, device
    )
    print(f"actual trained vocab size: {cfg.vocab_size}")
    print(f"\n=== base autoencoder: {model.num_parameters():,} params ===")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = random.Random(0)

    print(f"\n=== phase 1/4: reconstruction training ({args.recon_steps} steps) ===")
    recon_losses = phase_reconstruction(
        model, tok, cfg, texts, opt, args.recon_steps, args.batch_size, args.seq_len, device, rng
    )

    print("\n=== phase 2/4: contrastive check (two noise views of the same sentences) ===")
    same_pair_loss, shuffled_loss = phase_contrastive(model, tok, cfg, texts, args.seq_len, device, rng)
    print(f"  contrastive loss, same underlying sentence (2 noise views): {same_pair_loss:.4f}")
    print(f"  contrastive loss, mismatched/shuffled pairs:                {shuffled_loss:.4f}")
    print("  (lower = more similar; same-sentence pairs should score lower than shuffled ones)")

    print(f"\n=== phase 3/4: classification head ({args.classifier_steps} steps, {len(categories)}-way) ===")
    clf, clf_losses, clf_accs, full_acc = phase_classification(
        model, tok, cfg, texts, labels, categories, 1e-3, args.classifier_steps, args.batch_size, args.seq_len, device, rng
    )
    print(f"  final accuracy over the full (training) corpus: {full_acc:.2%}")

    print("\n=== phase 4/4: retrieval + anomaly detection ===")
    corpus_pooled, top_idx, top_scores = phase_retrieval(model, tok, cfg, texts, labels, args.seq_len, device, query_idx=0, k=4)
    print(f"\n  retrieval query: {texts[0]}  [{labels[0]}]")
    for i, score in zip(top_idx, top_scores):
        marker = " (self)" if i == 0 else ""
        print(f"    {score:.3f}  [{labels[i]}] {texts[i]}{marker}")

    anomaly_scores = phase_anomaly(model, tok, cfg, texts, args.seq_len, device, rng)
    for label, (text, score) in anomaly_scores.items():
        print(f"\n  anomaly score ({label.replace('_', '-')}): {score:.3f}   \"{text}\"")

    save_all_plots(
        args.docs_dir, recon_losses, [same_pair_loss, shuffled_loss], clf_losses, clf_accs,
        corpus_pooled, labels, categories, anomaly_scores, model.num_parameters(), device,
    )

    print("\n=== done: reconstruction, contrastive, classification, retrieval, and anomaly detection all exercised on the same latent Z ===")


if __name__ == "__main__":
    main()
