import torch
import torch.nn as nn
import torch.nn.functional as F


def pool_latent(z: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
    """Mean-pool the encoder's latent Z over valid (non-pad) positions into one vector per example."""
    if padding_mask is None:
        return z.mean(dim=1)
    mask = padding_mask.unsqueeze(-1).float()
    return (z * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Pull two latent views of the same underlying example together (1 - cosine similarity)."""
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    return (1.0 - (z1 * z2).sum(dim=-1)).mean()


class ClassificationHead(nn.Module):
    """Attaches to the pooled latent Z to predict a downstream label (e.g. topic category)."""

    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.proj = nn.Linear(d_model, num_classes)

    def forward(self, pooled_z: torch.Tensor) -> torch.Tensor:
        return self.proj(pooled_z)


def nearest_neighbors(query_embedding: torch.Tensor, corpus_embeddings: torch.Tensor, k: int = 3):
    """Cosine-similarity retrieval: rank corpus_embeddings by similarity to query_embedding.

    Returns (indices, scores), both length k, scores in descending order.
    """
    q = F.normalize(query_embedding, dim=-1)
    c = F.normalize(corpus_embeddings, dim=-1)
    scores = c @ q
    k = min(k, scores.shape[0])
    top_scores, top_indices = torch.topk(scores, k)
    return top_indices, top_scores


@torch.no_grad()
def reconstruction_error(model, target_ids: torch.Tensor, noisy_ids: torch.Tensor,
                          decoder_input_ids: torch.Tensor, pad_id: int,
                          encoder_padding_mask: torch.Tensor = None) -> torch.Tensor:
    """Per-example mean cross-entropy under the model -- higher means harder to reconstruct,
    i.e. more anomalous relative to what the model learned during training."""
    model.eval()
    logits = model(noisy_ids, decoder_input_ids, encoder_padding_mask)
    per_token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1),
        reduction="none", ignore_index=pad_id,
    ).view(target_ids.shape)

    valid = (target_ids != pad_id).float()
    return (per_token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
