"""Classification metrics: accuracy, F1, precision, recall, AUC-ROC, per-class."""
from typing import Dict

import torch
import numpy as np

from .embedding_quality import compute_embedding_quality


def compute_embedding_metrics(embeddings: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Embedding quality, legacy key names."""
    quality = compute_embedding_quality(embeddings, labels)
    return {
        'intra_class_sim': quality['intra_class_sim'],
        'inter_class_sim': quality['inter_class_sim'],
        'sim_gap': quality['separation_gap'],
        'centroid_similarity': quality['mean_centroid_dist'],
    }


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy."""
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def compute_per_class_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> Dict[str, float]:
    """Per-class accuracy + macro_avg."""
    preds = logits.argmax(dim=-1)
    correct = preds == labels

    per_class_acc = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            per_class_acc[f"class_{c}"] = correct[mask].float().mean().item()
        else:
            per_class_acc[f"class_{c}"] = 0.0

    valid_classes = [v for v in per_class_acc.values() if v > 0]
    per_class_acc["macro_avg"] = np.mean(valid_classes) if valid_classes else 0.0

    return per_class_acc


def compute_full_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> Dict[str, float]:
    """Accuracy, macro P/R/F1, macro AUC-ROC, per-class P/R/F1."""
    preds = logits.argmax(dim=-1)
    preds_np = preds.numpy()
    labels_np = labels.numpy()

    precisions, recalls, f1s = [], [], []
    per_class = {}

    for c in range(num_classes):
        tp = ((preds_np == c) & (labels_np == c)).sum()
        fp = ((preds_np == c) & (labels_np != c)).sum()
        fn = ((preds_np != c) & (labels_np == c)).sum()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[f"class_{c}_precision"] = prec
        per_class[f"class_{c}_recall"] = rec
        per_class[f"class_{c}_f1"] = f1

        # macro over present classes only
        if (labels_np == c).sum() > 0:
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

    metrics = {
        'accuracy': (preds_np == labels_np).mean(),
        'macro_precision': np.mean(precisions) if precisions else 0.0,
        'macro_recall': np.mean(recalls) if recalls else 0.0,
        'macro_f1': np.mean(f1s) if f1s else 0.0,
    }

    # one-vs-rest macro AUC
    probs = torch.softmax(logits, dim=-1).numpy()
    auc_scores = []
    for c in range(num_classes):
        binary_labels = (labels_np == c).astype(np.float64)
        if binary_labels.sum() == 0 or binary_labels.sum() == len(binary_labels):
            continue
        scores = probs[:, c]
        auc_scores.append(_auc_binary(binary_labels, scores))

    metrics['macro_auc_roc'] = np.mean(auc_scores) if auc_scores else 0.0
    metrics.update(per_class)
    return metrics


def _auc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    """Binary AUC, trapezoidal."""
    fpr, tpr, auc = _roc_curve_binary(labels, scores)
    return auc


def _roc_curve_binary(labels: np.ndarray, scores: np.ndarray):
    """(fpr, tpr, auc) via descending-score sweep, (0, 0) anchor prepended."""
    desc_idx = np.argsort(-scores)
    labels_sorted = labels[desc_idx]

    tps = np.cumsum(labels_sorted)
    fps = np.cumsum(1 - labels_sorted)

    tpr = tps / tps[-1] if tps[-1] > 0 else tps
    fpr = fps / fps[-1] if fps[-1] > 0 else fps

    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    _trapz = getattr(np, 'trapezoid', None) or np.trapz
    auc = _trapz(tpr, fpr)
    return fpr, tpr, auc


def compute_extended_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> Dict[str, object]:
    """probs (N, C) softmax (not re-normalised), labels (N,).

    Returns nll, brier, cohen_kappa, mcc (Gorodkin), macro_f1, macro_auc_roc,
    confusion_matrix (C, C) rows=true, roc_per_class [{class, fpr, tpr, auc}].
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    N = probs.shape[0]
    eps = 1e-12

    if N == 0:
        return {
            'nll': 0.0, 'brier': 0.0, 'cohen_kappa': 0.0, 'mcc': 0.0,
            'macro_f1': 0.0, 'macro_auc_roc': 0.0,
            'confusion_matrix': [[0] * num_classes for _ in range(num_classes)],
            'roc_per_class': [],
        }

    preds = probs.argmax(axis=-1)

    true_p = np.clip(probs[np.arange(N), labels], eps, None)
    nll = float(-np.log(true_p).mean())

    # Brier: sum over classes, mean over samples
    onehot = np.zeros_like(probs)
    onehot[np.arange(N), labels] = 1.0
    brier = float(((probs - onehot) ** 2).sum(axis=-1).mean())

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (labels, preds), 1)

    # same definitions as compute_full_metrics
    f1s = []
    for c in range(num_classes):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - cm[c, c])
        fn = float(cm[c, :].sum() - cm[c, c])
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if (labels == c).sum() > 0:
            f1s.append(
                2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            )
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    po = float((preds == labels).mean())
    p_true = cm.sum(axis=1) / N
    p_pred = cm.sum(axis=0) / N
    pe = float((p_true * p_pred).sum())
    cohen_kappa = (po - pe) / (1 - pe) if pe < 1.0 else 0.0

    # Gorodkin MCC
    t = cm.sum(axis=1).astype(np.float64)
    p = cm.sum(axis=0).astype(np.float64)
    s = float(N)
    correct = float(np.trace(cm))
    denom = (s * s - float((p * p).sum())) * (s * s - float((t * t).sum()))
    if denom <= 0:
        mcc = 0.0
    else:
        mcc = (correct * s - float((t * p).sum())) / np.sqrt(denom)

    roc_per_class = []
    auc_scores = []
    for c in range(num_classes):
        binary = (labels == c).astype(np.float64)
        if binary.sum() == 0 or binary.sum() == N:
            roc_per_class.append({'class': c, 'fpr': [], 'tpr': [], 'auc': None})
            continue
        scores = probs[:, c]
        fpr, tpr, auc = _roc_curve_binary(binary, scores)
        roc_per_class.append({
            'class': c,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': float(auc),
        })
        auc_scores.append(auc)
    macro_auc = float(np.mean(auc_scores)) if auc_scores else 0.0

    return {
        'nll': nll,
        'brier': brier,
        'cohen_kappa': float(cohen_kappa),
        'mcc': float(mcc),
        'macro_f1': macro_f1,
        'macro_auc_roc': macro_auc,
        'confusion_matrix': cm.tolist(),
        'roc_per_class': roc_per_class,
    }
