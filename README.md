# Transformer Autoencoder (~1.04B parameters)

A denoising transformer autoencoder (BART/T5-style encoder-decoder), matching:

```
Raw Text -> Tokenizer (32k vocab) -> Noise (mask/delete/replace)
  -> Encoder (RMSNorm -> Self-Attn -> FFN, x15, d_model=1536, 24 heads) -> Latent Z
  -> Decoder (Self-Attn -> Cross-Attn(Z) -> FFN, x15) -> Output Projection (32k)
  -> Reconstructed Text -> Cross-Entropy Loss (original vs reconstructed)
```

Verified parameter count (`python inspect_model.py`): **1,041,747,456** (~1.042B), with
tied input/output embeddings. Confirmed with an actual forward pass, not just arithmetic.

## Honest scope

This is real, runnable architecture + training code, not a pretrained model. Training
a 1B-parameter transformer to convergence needs a corpus of many billions of tokens and
days of multi-GPU compute -- well beyond what `train.py` does by default. `train.py` runs
a small number of steps on a small generated corpus to prove the wiring (noising, encoder,
decoder, tied-embedding projection, loss) is correct. Point it at more data and a GPU to scale.

## Layout

- `model/` -- the architecture (`config.py`, `layers.py`, `encoder.py`, `decoder.py`, `autoencoder.py`)
- `noise.py` -- mask/delete/replace token corruption
- `tokenizer/train_tokenizer.py` -- trains a byte-level BPE tokenizer to vocab_size=32000
- `data/generate_corpus.py` -- generates training text via the OpenAI and Claude APIs
- `train.py` -- demo training loop (noise -> encode -> decode -> cross-entropy -> AdamW)
- `inspect_model.py` -- parameter-count + forward-pass sanity check
- `notebooks/` -- the same pipeline as interactive notebooks (see below)

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY and ANTHROPIC_API_KEY, then `set -a; source .env; set +a`
```

Never hardcode API keys in source files -- both scripts read them from the environment only.

## Run order

```bash
python inspect_model.py                                   # sanity check: ~1.04B params, forward pass ok

python data/generate_corpus.py --provider both --num-samples 200 --out data/corpus.txt
python tokenizer/train_tokenizer.py --corpus data/corpus.txt --out-dir tokenizer/vocab

python train.py --corpus data/corpus.txt --tokenizer-dir tokenizer/vocab --steps 50
```

## Notebooks

`notebooks/` walks through the same pipeline interactively, in order:

1. `01_inspect_model.ipynb` -- build the full ~1.04B-param model, confirm the count and a forward pass
2. `02_generate_corpus.ipynb` -- sample text from the OpenAI and Claude APIs into `data/corpus.txt`
3. `03_train_tokenizer.ipynb` -- train the 32k-vocab BPE tokenizer and round-trip a sentence
4. `04_train_autoencoder.ipynb` -- run the noise/encode/decode/loss training loop and plot the loss curve

Notebook 4 defaults to a small `DEMO_CONFIG` (a few million params) so it runs on a laptop CPU;
flip `USE_FULL_SIZE_MODEL = True` once you're on hardware that can hold the real ~1.04B-param
model (see the RAM/GPU notes below).

## Scaling to a real pretraining run

- Swap `data/generate_corpus.py`'s output for a real corpus (web text, books, code, etc.) --
  API-generated text alone is not a substitute for a pretraining-scale dataset.
- Increase `--steps` by orders of magnitude and use a learning-rate schedule (warmup + decay).
- Move training to multi-GPU (`torch.distributed` / FSDP) -- CPU training in this environment
  is only suitable for the wiring check above.
- Consider mixed precision (`torch.autocast`) and gradient checkpointing to fit the ~1B-param
  model's activations in memory at larger batch/sequence sizes.
