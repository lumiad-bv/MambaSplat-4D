"""Embedding quality metrics for spatial pre-training validation."""
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_intra_class_similarity(embeddings: Tensor, labels: Tensor) -> float:
    """Mean within-class cosine similarity, averaged over classes."""
    emb = F.normalize(embeddings, dim=-1)
    classes = labels.unique()
    sims = []
    for c in classes:
        mask = labels == c
        if mask.sum() < 2:
            continue
        class_emb = emb[mask]
        cos = class_emb @ class_emb.T
        n = cos.shape[0]
        triu_mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=cos.device), diagonal=1)
        sims.append(cos[triu_mask].mean().item())
    return sum(sims) / len(sims) if sims else 0.0


def compute_inter_class_similarity(embeddings: Tensor, labels: Tensor) -> float:
    """Mean cosine similarity between class centroids."""
    emb = F.normalize(embeddings, dim=-1)
    classes = labels.unique()
    if len(classes) < 2:
        return 0.0

    centroids = []
    for c in classes:
        centroids.append(emb[labels == c].mean(dim=0))
    centroids = F.normalize(torch.stack(centroids), dim=-1)

    cos = centroids @ centroids.T
    n = cos.shape[0]
    triu_mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=cos.device), diagonal=1)
    return cos[triu_mask].mean().item()


def compute_separation_gap(embeddings: Tensor, labels: Tensor) -> float:
    """intra - inter; expect > 0.3."""
    intra = compute_intra_class_similarity(embeddings, labels)
    inter = compute_inter_class_similarity(embeddings, labels)
    return intra - inter


def compute_centroid_distances(embeddings: Tensor, labels: Tensor) -> Dict[str, float]:
    """Pairwise L2 between class centroids: mean/min/max."""
    emb = F.normalize(embeddings, dim=-1)
    classes = labels.unique().tolist()

    centroids = []
    for c in classes:
        centroids.append(emb[labels == c].mean(dim=0))
    centroids = torch.stack(centroids)

    dists = torch.cdist(centroids.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)
    n = len(classes)
    triu_mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=dists.device), diagonal=1)

    return {
        'mean_centroid_dist': dists[triu_mask].mean().item(),
        'min_centroid_dist': dists[triu_mask].min().item(),
        'max_centroid_dist': dists[triu_mask].max().item(),
    }


def compute_embedding_quality(embeddings: Tensor, labels: Tensor) -> Dict[str, float]:
    """intra, inter, separation_gap, centroid distances."""
    intra = compute_intra_class_similarity(embeddings, labels)
    inter = compute_inter_class_similarity(embeddings, labels)
    centroid = compute_centroid_distances(embeddings, labels)
    return {
        'intra_class_sim': intra,
        'inter_class_sim': inter,
        'separation_gap': intra - inter,
        **centroid,
    }
