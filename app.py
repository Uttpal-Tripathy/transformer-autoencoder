"""Real-time testing app for the RAE-1B transformer autoencoder.

Serves the same reconstruction / contrastive / classification / retrieval / anomaly
capabilities demonstrated in notebooks/04_train_autoencoder.ipynb and
notebooks/05_downstream_demo.ipynb as an interactive Gradio app, built on the trained
bundle produced by train_and_export.py.

Run:
    python train_and_export.py   # once, to train + export the bundle (see that file)
    python app.py                # serves on 0.0.0.0:$PORT (default 7860)
"""
import os
from pathlib import Path

import gradio as gr
import torch
from tokenizers import ByteLevelBPETokenizer

from model.config import ModelConfig
from model.autoencoder import TransformerAutoencoder
from model.downstream import pool_latent, ClassificationHead, nearest_neighbors, reconstruction_error
from noise import corrupt
from run_demo import encode_batch, shift_right, decode_ids

BUNDLE_PATH = os.environ.get("APP_BUNDLE", "checkpoints/app_bundle.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_bundle(bundle_path: str):
    if not Path(bundle_path).exists():
        raise SystemExit(
            f"No trained bundle at {bundle_path}. Run `python train_and_export.py` first "
            "(the Dockerfile does this automatically at image build time)."
        )
    bundle = torch.load(bundle_path, map_location=DEVICE, weights_only=False)
    cfg: ModelConfig = bundle["config"]

    tok = ByteLevelBPETokenizer(
        f"{bundle['tokenizer_dir']}/vocab.json", f"{bundle['tokenizer_dir']}/merges.txt"
    )

    model = TransformerAutoencoder(cfg).to(DEVICE)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()

    clf = ClassificationHead(cfg.d_model, len(bundle["categories"])).to(DEVICE)
    clf.load_state_dict(bundle["classifier_state_dict"])
    clf.eval()

    return {
        "tok": tok,
        "cfg": cfg,
        "model": model,
        "clf": clf,
        "categories": bundle["categories"],
        "texts": bundle["texts"],
        "labels": bundle["labels"],
        "corpus_embeddings": bundle["corpus_embeddings"].to(DEVICE),
        "seq_len": bundle["seq_len"],
        "anomaly_threshold": bundle["anomaly_threshold"],
        "anomaly_mean": bundle["anomaly_mean"],
        "anomaly_std": bundle["anomaly_std"],
        "train_accuracy": bundle.get("train_accuracy"),
        "holdout_accuracy": bundle.get("holdout_accuracy"),
        "holdout_size": bundle.get("holdout_size"),
    }


STATE = load_bundle(BUNDLE_PATH)


def _encode_to_latent(text: str):
    tok, cfg, model, seq_len = STATE["tok"], STATE["cfg"], STATE["model"], STATE["seq_len"]
    ids = encode_batch(tok, cfg, [text], seq_len).to(DEVICE)
    mask = ids != cfg.pad_id
    with torch.no_grad():
        z = model.encoder(model.embed(ids), mask)
    return pool_latent(z, mask)


def _corrupt_and_generate(text: str, max_new_tokens: int = 63):
    tok, cfg, model, seq_len = STATE["tok"], STATE["cfg"], STATE["model"], STATE["seq_len"]

    target_ids = encode_batch(tok, cfg, [text], seq_len).to(DEVICE)
    row = [t for t in target_ids[0].tolist() if t != cfg.pad_id]
    noised = corrupt(row, cfg)[:seq_len]
    noised = noised + [cfg.pad_id] * (seq_len - len(noised))
    noisy_ids = torch.tensor([noised], dtype=torch.long, device=DEVICE)
    encoder_padding_mask = noisy_ids != cfg.pad_id

    with torch.no_grad():
        memory = model.encoder(model.embed(noisy_ids), encoder_padding_mask)
        generated = torch.full((1, 1), cfg.bos_id, dtype=torch.long, device=DEVICE)
        recent = []
        for _ in range(max_new_tokens):
            hidden = model.decoder(model.embed(generated), memory, encoder_padding_mask)
            next_logits = model.project_to_vocab(hidden[:, -1:]).clone()
            for tok_id in set(recent[-4:]):
                next_logits[0, 0, tok_id] -= 8.0
            next_token = next_logits.argmax(dim=-1)
            recent.append(next_token.item())
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == cfg.eos_id:
                break

    corrupted_text = decode_ids(tok, noised)
    reconstructed_text = decode_ids(tok, generated[0].tolist())
    return corrupted_text, reconstructed_text


def reconstruct_fn(text: str):
    if not text or not text.strip():
        return "", ""
    return _corrupt_and_generate(text.strip())


def classify_fn(text: str):
    if not text or not text.strip():
        return {}
    pooled = _encode_to_latent(text.strip())
    with torch.no_grad():
        probs = torch.softmax(STATE["clf"](pooled), dim=-1)[0]
    return {cat: float(p) for cat, p in zip(STATE["categories"], probs.tolist())}


def retrieve_fn(text: str, k: int):
    if not text or not text.strip():
        return []
    query_pooled = _encode_to_latent(text.strip())
    top_idx, top_scores = nearest_neighbors(query_pooled[0], STATE["corpus_embeddings"], k=int(k))
    rows = []
    for i, score in zip(top_idx.tolist(), top_scores.tolist()):
        rows.append([f"{score:.3f}", STATE["labels"][i], STATE["texts"][i]])
    return rows


def anomaly_fn(text: str):
    if not text or not text.strip():
        return 0.0, "-"
    tok, cfg, model, seq_len = STATE["tok"], STATE["cfg"], STATE["model"], STATE["seq_len"]
    target_ids = encode_batch(tok, cfg, [text.strip()], seq_len).to(DEVICE)
    row = [t for t in target_ids[0].tolist() if t != cfg.pad_id]
    noised = corrupt(row, cfg)[:seq_len]
    noised = noised + [cfg.pad_id] * (seq_len - len(noised))
    noisy_ids = torch.tensor([noised], dtype=torch.long, device=DEVICE)
    decoder_input_ids = shift_right(target_ids, cfg.bos_id)
    score = reconstruction_error(
        model, target_ids, noisy_ids, decoder_input_ids, cfg.pad_id, noisy_ids != cfg.pad_id
    ).item()
    verdict = "ANOMALOUS" if score > STATE["anomaly_threshold"] else "normal"
    detail = (
        f"{verdict}  (score {score:.2f}, corpus baseline {STATE['anomaly_mean']:.2f} "
        f"+/- {STATE['anomaly_std']:.2f}, threshold {STATE['anomaly_threshold']:.2f})"
    )
    return round(score, 3), detail


ARCHITECTURE_INFO = f"""
### Model

- Encoder-decoder transformer autoencoder, **{STATE['model'].num_parameters():,} parameters**
- Trained on `data/sample_corpus.tsv`: {len(STATE['texts'])} sentences across {len(STATE['categories'])} topics: {', '.join(STATE['categories'])}
- Device serving this app: **{DEVICE}**

### What each tab does

All five tabs read from the **same encoder and the same pooled latent Z** -- this app is
the live version of `notebooks/05_downstream_demo.ipynb` and `run_downstream_demo.py`.

- **Reconstruct**: corrupts your text (mask/delete/replace) and decodes it back through the trained autoencoder
- **Classify**: predicts a topic category from the pooled latent Z
- **Retrieve**: nearest-neighbor search over the training corpus by latent similarity
- **Anomaly**: reconstruction-error score against a threshold calibrated on
  {STATE['holdout_size']} held-out sentences the model never trained on (mean
  {STATE['anomaly_mean']:.2f} +/- {STATE['anomaly_std']:.2f}, threshold
  {STATE['anomaly_threshold']:.2f}) -- calibrating on training sentences instead would
  have made almost *any* new phrasing look anomalous, since the model has nearly
  memorized its ~120-sentence training set

### Honest scope

- Classifier accuracy: **{STATE['train_accuracy']:.1%}** on the sentences it trained on
  vs. **{STATE['holdout_accuracy']:.1%}** on held-out sentences it never saw -- that gap
  is the real generalization picture, not the inflated training number alone.
- This is a demonstration-scale model (a small corpus, ~45M parameters), not a production
  NLP system -- see the main [README](https://github.com/Uttpal-Tripathy/transformer-autoencoder)
  for more real, unfiltered numbers from training this exact model.
"""


with gr.Blocks(title="RAE-1B Live Testing") as demo:
    gr.Markdown("# RAE-1B: Real-Time Testing App")
    gr.Markdown(
        "Type text into any tab below to test the trained transformer autoencoder live. "
        "**Classify** and **Anomaly** update as you type; **Reconstruct** and **Retrieve** "
        "update on submit (they involve randized noise / a larger search, so a button keeps "
        "results reproducible per click)."
    )

    with gr.Tab("Reconstruct"):
        recon_in = gr.Textbox(label="Input text", placeholder="Type a sentence...", lines=2)
        recon_btn = gr.Button("Corrupt & Reconstruct", variant="primary")
        with gr.Row():
            corrupted_out = gr.Textbox(label="Corrupted (mask/delete/replace)", interactive=False)
            recon_out = gr.Textbox(label="Model's reconstruction", interactive=False)
        recon_btn.click(reconstruct_fn, inputs=recon_in, outputs=[corrupted_out, recon_out])
        recon_in.submit(reconstruct_fn, inputs=recon_in, outputs=[corrupted_out, recon_out])

    with gr.Tab("Classify"):
        clf_in = gr.Textbox(label="Input text", placeholder="Type a sentence...", lines=2)
        clf_out = gr.Label(label="Predicted topic", num_top_classes=len(STATE["categories"]))
        clf_in.change(classify_fn, inputs=clf_in, outputs=clf_out)

    with gr.Tab("Retrieve"):
        ret_in = gr.Textbox(label="Query text", placeholder="Type a sentence...", lines=2)
        ret_k = gr.Slider(1, 10, value=4, step=1, label="Number of neighbors (k)")
        ret_btn = gr.Button("Search", variant="primary")
        ret_out = gr.Dataframe(headers=["score", "topic", "text"], label="Nearest neighbors")
        ret_btn.click(retrieve_fn, inputs=[ret_in, ret_k], outputs=ret_out)
        ret_in.submit(retrieve_fn, inputs=[ret_in, ret_k], outputs=ret_out)

    with gr.Tab("Anomaly"):
        gr.Markdown(
            "Threshold is calibrated on held-out sentences the model never trained on "
            "(see the Info tab), not on training data -- otherwise almost any new phrasing "
            "would look anomalous. **Honest limitation:** with only ~100 training sentences, "
            "this mainly distinguishes text at or near what the model has seen from anything "
            "unfamiliar -- it does not reliably separate grammatical-but-novel text from "
            "nonsense (both tend to score as unfamiliar). It reliably catches near-duplicates "
            "of training text as normal; treat anything past that as a rough signal, not a "
            "precise semantic anomaly detector."
        )
        an_in = gr.Textbox(label="Input text", placeholder="Type a sentence...", lines=2)
        an_score = gr.Number(label="Reconstruction-error anomaly score")
        an_verdict = gr.Textbox(label="Verdict", interactive=False)
        an_in.change(anomaly_fn, inputs=an_in, outputs=[an_score, an_verdict])

    with gr.Tab("Info"):
        gr.Markdown(ARCHITECTURE_INFO)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.queue().launch(server_name="0.0.0.0", server_port=port)
