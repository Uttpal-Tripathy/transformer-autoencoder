"""One-time training + export step for the real-time testing app.

Trains the base autoencoder and the classification head on a training split of
data/sample_corpus.tsv, computes retrieval embeddings over the full corpus, and
calibrates the anomaly-score threshold on a *held-out* split the model never trained
on. Calibrating on held-out data matters: with only ~120 sentences the model nearly
memorizes its training set (near-zero reconstruction error), so a threshold fit on
training sentences flags almost any new phrasing -- even a harmless paraphrase -- as
"anomalous". A held-out split's reconstruction error reflects genuine generalization
error instead, giving a threshold that's actually usable.

Run this once before `python app.py` -- the Dockerfile also runs it at image build
time, so a deployed container starts with weights already trained instead of
training on every boot.

--recon-steps defaults to 600, not more: pushing past this mainly increases the gap
between train and held-out performance (tried 1500 -- reconstruction loss looked
better, but classifier held-out accuracy dropped and the held-out anomaly baseline
score climbed, i.e. purely memorization, not learning). Even at 600 steps, the
anomaly detector mainly distinguishes "text at or near what the model has seen"
from "anything unfamiliar" -- it does not reliably separate grammatical-but-novel
text from nonsense, since ~100 sentences isn't enough to learn general language
structure. See the Anomaly tab / README for the honest characterization.
"""
import argparse
import random
from pathlib import Path

import torch

from run_downstream_demo import (
    load_labeled_corpus, build_model_and_tokenizer, encode_to_latent,
    phase_reconstruction, phase_classification,
)
from model.downstream import reconstruction_error
from noise import corrupt
from run_demo import encode_batch, shift_right


def split_train_holdout(labels, texts, holdout_frac, seed):
    idx = list(range(len(texts)))
    random.Random(seed).shuffle(idx)
    n_holdout = max(1, int(len(idx) * holdout_frac))
    holdout_idx, train_idx = set(idx[:n_holdout]), idx[n_holdout:]
    train = ([labels[i] for i in train_idx], [texts[i] for i in train_idx])
    holdout = ([labels[i] for i in sorted(holdout_idx)], [texts[i] for i in sorted(holdout_idx)])
    return train, holdout


def reconstruction_scores(model, tok, cfg, texts, seq_len, device, rng):
    scores = []
    for text in texts:
        target_ids = encode_batch(tok, cfg, [text], seq_len).to(device)
        row = [t for t in target_ids[0].tolist() if t != cfg.pad_id]
        noised = corrupt(row, cfg, rng=rng)[:seq_len]
        noised = noised + [cfg.pad_id] * (seq_len - len(noised))
        noisy_ids = torch.tensor([noised], dtype=torch.long, device=device)
        decoder_input_ids = shift_right(target_ids, cfg.bos_id)
        score = reconstruction_error(model, target_ids, noisy_ids, decoder_input_ids, cfg.pad_id, noisy_ids != cfg.pad_id)
        scores.append(score.item())
    return torch.tensor(scores)


def classification_accuracy(model, clf, tok, cfg, texts, labels, categories, seq_len, device):
    from model.downstream import pool_latent
    label_to_idx = {c: i for i, c in enumerate(categories)}
    label_ids = torch.tensor([label_to_idx[l] for l in labels], dtype=torch.long, device=device)
    ids = encode_batch(tok, cfg, texts, seq_len).to(device)
    mask = ids != cfg.pad_id
    with torch.no_grad():
        pooled = pool_latent(model.encoder(model.embed(ids), mask), mask)
        acc = (clf(pooled).argmax(-1) == label_ids).float().mean().item()
    return acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data/sample_corpus.tsv")
    p.add_argument("--tokenizer-dir", default="tokenizer/vocab")
    p.add_argument("--vocab-size", type=int, default=4000)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--recon-steps", type=int, default=600)
    p.add_argument("--classifier-steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--holdout-frac", type=float, default=0.15)
    p.add_argument("--bundle", default="checkpoints/app_bundle.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = args.device
    print(f"device: {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    all_labels, all_texts = load_labeled_corpus(args.corpus)
    categories = sorted(set(all_labels))
    print(f"corpus: {len(all_texts)} lines, {len(categories)} categories: {categories}")

    (train_labels, train_texts), (holdout_labels, holdout_texts) = split_train_holdout(
        all_labels, all_texts, args.holdout_frac, seed=0
    )
    print(f"train: {len(train_texts)} sentences, held out (never trained on): {len(holdout_texts)} sentences")

    plain_corpus_path = Path("data/sample_corpus.txt")
    plain_corpus_path.write_text("\n".join(all_texts) + "\n", encoding="utf-8")

    tok, cfg, model = build_model_and_tokenizer(
        str(plain_corpus_path), args.tokenizer_dir, args.vocab_size, args.seq_len, device
    )
    print(f"model parameters: {model.num_parameters():,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = random.Random(0)

    print(f"training reconstruction on the train split ({args.recon_steps} steps)...")
    phase_reconstruction(model, tok, cfg, train_texts, opt, args.recon_steps, args.batch_size, args.seq_len, device, rng)

    print(f"training classifier on the train split ({args.classifier_steps} steps)...")
    clf, _, _, train_acc = phase_classification(
        model, tok, cfg, train_texts, train_labels, categories, 1e-3, args.classifier_steps, args.batch_size, args.seq_len, device, rng
    )
    holdout_acc = classification_accuracy(model, clf, tok, cfg, holdout_texts, holdout_labels, categories, args.seq_len, device)
    print(f"classifier accuracy -- train split: {train_acc:.2%}  held-out split: {holdout_acc:.2%}")

    print("computing corpus embeddings for retrieval (full corpus)...")
    corpus_pooled, _, _ = encode_to_latent(model, tok, cfg, all_texts, args.seq_len, device)

    print("calibrating anomaly-score threshold on the held-out split...")
    holdout_scores = reconstruction_scores(model, tok, cfg, holdout_texts, args.seq_len, device, rng)
    anomaly_mean, anomaly_std = holdout_scores.mean().item(), holdout_scores.std().item()
    anomaly_threshold = anomaly_mean + 2 * anomaly_std
    print(f"anomaly baseline (held-out): mean={anomaly_mean:.3f} std={anomaly_std:.3f} threshold={anomaly_threshold:.3f}")

    train_scores = reconstruction_scores(model, tok, cfg, train_texts, args.seq_len, device, rng)
    print(f"  (for comparison, train-split scores: mean={train_scores.mean().item():.3f} "
          f"std={train_scores.std().item():.3f} -- much lower, since the model memorized these)")

    bundle_path = Path(args.bundle)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": cfg,
        "model_state_dict": model.state_dict(),
        "classifier_state_dict": clf.state_dict(),
        "categories": categories,
        "texts": all_texts,
        "labels": all_labels,
        "corpus_embeddings": corpus_pooled.cpu(),
        "tokenizer_dir": args.tokenizer_dir,
        "seq_len": args.seq_len,
        "anomaly_threshold": anomaly_threshold,
        "anomaly_mean": anomaly_mean,
        "anomaly_std": anomaly_std,
        "train_accuracy": train_acc,
        "holdout_accuracy": holdout_acc,
        "holdout_size": len(holdout_texts),
    }, bundle_path)
    print(f"saved bundle: {bundle_path}")


if __name__ == "__main__":
    main()
