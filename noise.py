import random
from typing import List

from model.config import ModelConfig


def corrupt(ids: List[int], cfg: ModelConfig, mask_prob: float = 0.15,
            delete_prob: float = 0.10, replace_prob: float = 0.10,
            rng: random.Random = None) -> List[int]:
    """Apply BART-style noise: each token is independently masked, deleted,
    replaced with a random vocab id, or left alone."""
    rng = rng or random.Random()
    out = []
    for tok in ids:
        if tok in (cfg.bos_id, cfg.eos_id, cfg.pad_id):
            out.append(tok)
            continue
        r = rng.random()
        if r < delete_prob:
            continue
        elif r < delete_prob + mask_prob:
            out.append(cfg.mask_id)
        elif r < delete_prob + mask_prob + replace_prob:
            out.append(rng.randrange(5, cfg.vocab_size))  # skip special ids 0-4
        else:
            out.append(tok)
    return out or [cfg.mask_id]
