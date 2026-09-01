"""Generate a training corpus by sampling text from the OpenAI and Claude APIs.

Both API keys are read from environment variables only -- never hardcode them:
    OPENAI_API_KEY
    ANTHROPIC_API_KEY

Usage:
    python data/generate_corpus.py --provider both --num-samples 200 --out data/corpus.txt
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


def gen_claude(client, topic: str, model: str) -> str:
    resp = client.messages.create(
        model=model,
        max_tokens=300,
        temperature=1.0,
        system="Write natural, varied English prose. No markdown, no lists.",
        messages=[{"role": "user", "content": f"Write a self-contained paragraph in the style of {topic}."}],
    )
    return "".join(block.text for block in resp.content if block.type == "text").strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["openai", "claude", "both"], default="both")
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--out", default="data/corpus.txt")
    p.add_argument("--openai-model", default="gpt-4.1-mini")
    p.add_argument("--claude-model", default="claude-sonnet-5")
    args = p.parse_args()

    openai_client = None
    claude_client = None

    if args.provider in ("openai", "both"):
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set in the environment.")
        from openai import OpenAI
        openai_client = OpenAI()

    if args.provider in ("claude", "both"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is not set in the environment.")
        from anthropic import Anthropic
        claude_client = Anthropic()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("a", encoding="utf-8") as f:
        while written < args.num_samples:
            topic = DEFAULT_TOPICS[written % len(DEFAULT_TOPICS)]
            use_openai = args.provider == "openai" or (args.provider == "both" and written % 2 == 0)
            try:
                if use_openai and openai_client is not None:
                    text = gen_openai(openai_client, topic, args.openai_model)
                elif claude_client is not None:
                    text = gen_claude(claude_client, topic, args.claude_model)
                else:
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
