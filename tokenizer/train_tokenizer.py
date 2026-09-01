import argparse
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<mask>"]  # ids 0-4, must match model/config.py


def train(corpus_path: str, out_dir: str, vocab_size: int = 32000):
    tok = ByteLevelBPETokenizer()
    tok.train(
        files=[corpus_path],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tok.save_model(out_dir)
    print(f"tokenizer trained on {corpus_path}, saved to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data/corpus.txt")
    p.add_argument("--out-dir", default="tokenizer/vocab")
    p.add_argument("--vocab-size", type=int, default=32000)
    args = p.parse_args()
    train(args.corpus, args.out_dir, args.vocab_size)
