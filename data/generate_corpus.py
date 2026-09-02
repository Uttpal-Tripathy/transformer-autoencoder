"""Generate a training corpus by sampling text from the OpenAI API.

The API key is read from an environment variable only -- never hardcode it:
    OPENAI_API_KEY

Usage:
    python data/generate_corpus.py --num-samples 200 --out data/corpus.txt
"""
import argparse
import os
import time
from pathlib import Path

DEFAULT_TOPICS = [
    "a short news report", "a technical explainer", "a product review",
    "a piece of dialogue between two people", "a recipe", "a travel diary entry",
    "a scientific abstract", "a historical summary", "an opinion essay",
    "instructions for a DIY project", "a movie synopsis", "a sports recap",
]


def gen_openai(client, topic: str, model: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Write natural, varied English prose. No markdown, no lists."},
            {"role": "user", "content": f"Write a self-contained paragraph in the style of {topic}."},
        ],
        max_tokens=300,
        temperature=1.0,
    )
    return resp.choices[0].message.content.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--out", default="data/corpus.txt")
    p.add_argument("--openai-model", default="gpt-4.1-mini")
    args = p.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set in the environment.")
    from openai import OpenAI
    openai_client = OpenAI()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("a", encoding="utf-8") as f:
        while written < args.num_samples:
            topic = DEFAULT_TOPICS[written % len(DEFAULT_TOPICS)]
            try:
                text = gen_openai(openai_client, topic, args.openai_model)
            except Exception as e:
                print(f"generation failed ({e}), retrying in 5s")
                time.sleep(5)
                continue

            text = text.replace("\n", " ").strip()
            if text:
                f.write(text + "\n")
                written += 1
                if written % 10 == 0:
                    print(f"{written}/{args.num_samples} samples written")

    print(f"done: {written} samples -> {out_path}")


if __name__ == "__main__":
    main()
