"""Supervised MambaSplat-4D trainer on AeroSplat-4D.

  python -m mambasplat4d.cli.train dataset=aerosplat4d_pre training=aerosplat4d_stage2
"""
import time
import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, StepLR
from tqdm import tqdm

from mambasplat4d import paths
from mambasplat4d.utils.seed import set_seed
from mambasplat4d.utils.logging import (
    init_wandb, finish_wandb, watch_model, save_checkpoint, TrainingLogger,
)
from mambasplat4d.models.pipeline import ClassificationPipeline
from mambasplat4d.schedulers.warmup_multistep import WarmupMultiStepLR
from mambasplat4d.datasets.isaacsim_pre import IsaacSimPreDataset
from mambasplat4d.datasets.collate import collate_sequences
from mambasplat4d.losses.classification import SupConLoss
from mambasplat4d.metrics.classification import compute_embedding_metrics, compute_full_metrics
from mambasplat4d.utils.feature_config import FEATURE_MODES
from mambasplat4d.cli._pipeline import get_device

ALL_GAUSSIAN_KEYS = ('position', 'quaternion', 'scale', 'opacity', 'sh_dc')


def get_feature_keys(cfg: DictConfig) -> tuple:
    """feature_mode -> Gaussian dict keys."""
    mode = cfg.dataset.get('feature_mode', 'X')
    if mode not in FEATURE_MODES:
        raise ValueError(f"Unknown feature_mode '{mode}'. Choose from: {list(FEATURE_MODES.keys())}")
    return FEATURE_MODES[mode]


def build_dataset(cfg: DictConfig, split: str):
    ds_cfg = cfg.dataset
    fmt = ds_cfg.format
    if fmt != 'isaacsim_pre':
        raise ValueError(
            f"Unsupported dataset format '{fmt}': this release trains on "
            "AeroSplat-4D only (dataset.format=isaacsim_pre)."
        )
    rot_mode = cfg.get('rotation', {}).get('train', 'none') if split == 'train' else 'none'
    aug = ds_cfg.augmentation
    return IsaacSimPreDataset(
        data_root=ds_cfg.data_root,
        split=split,
        num_points=ds_cfg.num_points,
        seq_len=ds_cfg.get('seq_len', None),
        feature_keys=get_feature_keys(cfg),
        frame_interval=ds_cfg.get('frame_interval', 1),
        n_versions=ds_cfg.get('n_versions', 25),
        train_stride=ds_cfg.get('train_stride', 1),
        train_rotation_mode=rot_mode,
        scale_range=tuple(aug.get('scale_range', [1.0, 1.0])),
        scale_per_axis=bool(aug.get('scale_per_axis', True)),
        version_per_frame=ds_cfg.get('version_per_frame', True),
        seed=cfg.seed,
    )


def to_device(batch, device):
    """Batch to device; returns (gaussian_data, mask, labels, T)."""
    gs_keys = [k for k in ALL_GAUSSIAN_KEYS if k in batch]
    B, T, N = batch['position'].shape[:3]
    gaussian_data = {k: batch[k].reshape(B * T, N, -1).to(device) for k in gs_keys}
    mask = batch['mask'].reshape(B * T, N).to(device)
    return gaussian_data, mask, batch['label'].to(device), T


def build_optimizer(pipeline, cfg: DictConfig):
    """Optimizer over pipeline.get_param_groups (differential LR)."""
    opt_cfg = cfg.training.optimizer
    param_groups = pipeline.get_param_groups(cfg)
    name = opt_cfg.get('name', 'adamw').lower()
    if name == 'sgd':
        return SGD(param_groups,
                   lr=opt_cfg.lr,
                   momentum=opt_cfg.get('momentum', 0.9),
                   weight_decay=opt_cfg.weight_decay)
    return AdamW(param_groups,
                 weight_decay=opt_cfg.weight_decay,
                 betas=tuple(opt_cfg.betas))


def build_scheduler(optimizer, cfg: DictConfig):
    """LR scheduler: cosine (default), step, or warmup multistep."""
    sched_cfg = cfg.training.scheduler
    name = sched_cfg.get('name', 'cosine').lower()

    if name == 'step':
        return StepLR(optimizer,
                      step_size=sched_cfg.get('step_size', 20),
                      gamma=sched_cfg.get('gamma', 0.7))

    if name == 'multistep':
        return WarmupMultiStepLR(
            optimizer,
            milestones=list(sched_cfg.lr_milestones),
            gamma=sched_cfg.get('lr_gamma', 0.1),
            warmup_iters=sched_cfg.get('lr_warmup_epochs', 10),
            warmup_factor=sched_cfg.get('warmup_factor', 1e-5),
        )

    warmup_epochs = sched_cfg.get('warmup_epochs', 0)
    total_epochs = cfg.training.epochs
    cosine = CosineAnnealingLR(optimizer,
                               T_max=max(total_epochs - warmup_epochs, 1),
                               eta_min=sched_cfg.eta_min)
    if warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=1e-2, total_iters=warmup_epochs)
        return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
    return cosine


class VNDGCNNLabelSmoothingLoss(nn.Module):
    """VN-DGCNN label smoothing: true = 1-eps, others = eps/(n_class-1).
    PyTorch built-in gives eps/n_class to every class incl. true."""

    def __init__(self, eps: float = 0.2):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        target = target.contiguous().view(-1)
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        return -(one_hot * log_prb).sum(dim=1).mean()


def compute_loss(output, labels, ce_fn, supcon_fn, loss_w, T):
    """Sequence CE plus optional weighted per-frame auxiliary terms."""
    terms = {'ce_loss': ce_fn(output['logits'], labels)}
    weights = {'ce_loss': loss_w.classification}
    if 'frame_logits' in output and loss_w.get('auxiliary_classification', 0) > 0:
        terms['aux_ce_loss'] = ce_fn(output['frame_logits'], labels.repeat_interleave(T))
        weights['aux_ce_loss'] = loss_w.auxiliary_classification
    if 'frame_projections' in output and loss_w.get('auxiliary_contrastive', 0) > 0:
        terms['aux_con_loss'] = supcon_fn(output['frame_projections'], labels.repeat_interleave(T))
        weights['aux_con_loss'] = loss_w.auxiliary_contrastive
    loss = sum(weights[k] * terms[k] for k in terms)
    return loss, {k: v.detach().item() for k, v in terms.items()}


def train_epoch(pipeline, loader, optimizer, ce_fn, supcon_fn, loss_w, device,
                accum_steps, grad_clip):
    """One training epoch."""
    pipeline.train()
    total_loss, correct, total = 0.0, 0, 0
    component_sums = {}
    optimizer.zero_grad()

    pbar = tqdm(loader, desc="train", leave=False, dynamic_ncols=True)
    for step, batch in enumerate(pbar):
        gaussian_data, mask, labels, T = to_device(batch, device)
        output = pipeline(gaussian_data, mask=mask, T=T)
        loss, components = compute_loss(output, labels, ce_fn, supcon_fn, loss_w, T)
        (loss / accum_steps).backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(pipeline.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        preds = output['logits'].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.shape[0]

        with torch.no_grad():
            components['ce_raw_loss'] = F.cross_entropy(output['logits'], labels).item()
        for k, v in components.items():
            component_sums[k] = component_sums.get(k, 0.0) + v

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

    n = max(len(loader), 1)
    loss_components = {k: v / n for k, v in component_sums.items()}
    return total_loss / n, correct / max(total, 1), loss_components


@torch.no_grad()
def validate(pipeline, loader, ce_fn, device):
    """Clip-level and video-level (softmax summed over clips) validation."""
    pipeline.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_embeddings, all_labels, all_logits = [], [], []
    video_prob, video_label = {}, {}

    for batch in tqdm(loader, desc="val", leave=False, dynamic_ncols=True):
        gaussian_data, mask, labels, T = to_device(batch, device)
        output = pipeline(gaussian_data, mask=mask, T=T)
        total_loss += ce_fn(output['logits'], labels).item()
        preds = output['logits'].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.shape[0]

        all_embeddings.append(output['embeddings'].cpu())
        all_labels.append(labels.cpu())
        all_logits.append(output['logits'].cpu())

        prob = F.softmax(output['logits'], dim=1).cpu()
        for i, vid in enumerate(batch['video_idx'].tolist()):
            if vid in video_prob:
                video_prob[vid] += prob[i]
            else:
                video_prob[vid] = prob[i].clone()
                video_label[vid] = int(labels[i])

    embeddings = torch.cat(all_embeddings, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    logits_cat = torch.cat(all_logits, dim=0)
    num_classes = logits_cat.shape[-1]

    metrics = compute_embedding_metrics(embeddings, labels_cat)
    cls_metrics = compute_full_metrics(logits_cat, labels_cat, num_classes)
    metrics['loss'] = total_loss / max(len(loader), 1)
    metrics['accuracy'] = correct / max(total, 1)
    for key in ('macro_precision', 'macro_recall', 'macro_f1', 'macro_auc_roc'):
        metrics[key] = cls_metrics[key]

    if video_prob:
        vids = sorted(video_prob)
        video_scores = torch.stack([video_prob[v] for v in vids])
        video_labels = torch.tensor([video_label[v] for v in vids])
        video_metrics = compute_full_metrics(video_scores, video_labels, num_classes)
        metrics['video_accuracy'] = float(video_metrics['accuracy'])
        metrics['video_macro_f1'] = float(video_metrics['macro_f1'])

    return metrics


def count_params_and_exit(cfg: DictConfig):
    output = cfg.params.get("output", None)
    if output is None:
        raise ValueError("+params.count_only=true requires +params.output=<path/to/params.json>")

    pipeline = ClassificationPipeline(cfg)
    total = sum(p.numel() for p in pipeline.parameters())
    trainable = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
    per_component = {
        name: sum(p.numel() for p in module.parameters() if p.requires_grad)
        for name, module in pipeline.named_children()
    }
    bridge_module = getattr(getattr(pipeline, "temporal_encoder", None), "bridge", None)
    bridge_meta = None
    if bridge_module is not None:
        per_component["temporal_encoder.bridge"] = sum(
            p.numel() for p in bridge_module.parameters() if p.requires_grad
        )
        bridge_meta = {"class": type(bridge_module).__name__}
    payload = {
        "total": total,
        "trainable": trainable,
        "per_component": per_component,
        "bridge": bridge_meta,
        "config": {
            "dataset": cfg.dataset.get("name", None),
            "feature_mode": cfg.dataset.get("feature_mode", None),
            "spatial": cfg.spatial.get("name", None),
            "temporal": cfg.temporal.get("name", None),
            "temporal_hidden_dim": cfg.temporal.get("hidden_dim", None),
            "temporal_n_layers": cfg.temporal.get("n_layers", None),
            "pos_encoding": cfg.temporal.get("pos_encoding", None),
            "translation_invariant": cfg.spatial.get("translation_invariant", None),
        },
    }
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    for k, v in per_component.items():
        print(f"  {k:<16s} {v:>12,}")
    print(f"Wrote {out_path}")


@hydra.main(version_base=None, config_path=paths.CONFIG_DIR, config_name="config")
def main(cfg: DictConfig):
    if hasattr(cfg.dataset, 'num_classes'):
        cfg.num_classes = cfg.dataset.num_classes

    if cfg.get("params", {}).get("count_only", False):
        count_params_and_exit(cfg)
        return

    heads_cfg = cfg.training.get('heads', {}) or {}
    freeze_cfg = cfg.training.get('freeze', {}) or {}
    if heads_cfg.get('frame_aux', False) and freeze_cfg.get('classifier', False):
        raise ValueError(
            "training.heads.frame_aux=true together with training.freeze.classifier=true "
            "(per-frame classification mode) is not supported in this release; "
            "set training.freeze.classifier=false to train the sequence classifier "
            "with per-frame auxiliary losses, or training.heads.frame_aux=false."
        )

    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("=" * 60)

    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / 'config.yaml', 'w') as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    set_seed(cfg.seed, deterministic=True)
    device = get_device(cfg)
    use_wandb = init_wandb(cfg)

    print(f"Device: {device} | Dataset: {cfg.dataset.name}")
    print(f"Batch size: {cfg.training.batch_size} | Epochs: {cfg.training.epochs}")
    accum_steps = cfg.training.get('gradient_accumulation_steps', 1)
    if accum_steps > 1:
        print(f"Gradient accumulation: {accum_steps} steps (effective batch={cfg.training.batch_size * accum_steps})")
    print(f"Seq len: {cfg.dataset.get('seq_len', 'full')}")
    print()

    train_dataset = build_dataset(cfg, 'train')
    val_dataset = build_dataset(cfg, 'val')
    print(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")

    nw = cfg.dataset.num_workers
    prefetch = cfg.dataset.get('prefetch_factor', 2)
    loader_kw = dict(
        num_workers=nw,
        collate_fn=collate_sequences,
        pin_memory=True,
        persistent_workers=nw > 0,
        multiprocessing_context="spawn" if nw > 0 else None,
        **({"prefetch_factor": prefetch} if nw > 0 else {}),
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size,
                              shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size,
                            shuffle=False, **loader_kw)

    log_model = cfg.wandb.get('log_model', False)
    logger = TrainingLogger(use_wandb=use_wandb, prefix="train")

    pipeline = ClassificationPipeline(cfg).to(device)
    n_params = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    rows = []
    for name, module in pipeline.named_children():
        n = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if n > 0:
            rows.append((name, n))
    if rows:
        w = max(len(name) for name, _ in rows)
        print(f"  {'component':<{w}}  {'params':>10}  {'%':>6}")
        print(f"  {'─' * w}  {'─' * 10}  {'─' * 6}")
        for name, n in rows:
            print(f"  {name:<{w}}  {n:>10,}  {n/n_params:>5.1%}")
        print()

    if cfg.hardware.get('compile', False):
        torch.set_float32_matmul_precision('high')
        compile_mode = cfg.hardware.get('compile_mode', 'default')
        pipeline = torch.compile(pipeline, mode=compile_mode)
        print(f"torch.compile enabled (mode='{compile_mode}', TF32 matmul)")

    if use_wandb:
        watch_model(pipeline, cfg)

    optimizer = build_optimizer(pipeline, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    _ls = cfg.training.loss.get('label_smoothing', 0.0)
    ce_fn = VNDGCNNLabelSmoothingLoss(eps=_ls) if _ls > 0 else nn.CrossEntropyLoss()
    con_cfg = cfg.training.get('contrastive', {}) or {}
    supcon_fn = SupConLoss(temperature=con_cfg.get('temperature', 0.1))
    loss_w = cfg.training.loss
    grad_clip = cfg.training.get('gradient_clip', 1.0)

    best_val_acc = 0.0
    final_train_acc = 0.0
    epochs_since_improve = 0
    _es = cfg.training.get('early_stopping', {}) or {}
    es_on = bool(_es.get('enabled', False))
    es_patience = int(_es.get('patience', 15))
    es_min_epochs = int(_es.get('min_epochs', 20))
    es_min_delta = float(_es.get('min_delta', 0.0))
    # Divergence guard: abort when train_loss frozen and train_acc collapsed;
    # avoids leaving a pre-divergence best_model.pt.
    _dg = cfg.training.get('divergence_guard', {}) or {}
    dg_on = bool(_dg.get('enabled', True))
    dg_window = int(_dg.get('window', 5))
    dg_tol = float(_dg.get('loss_tol', 1e-3))
    dg_frac = float(_dg.get('acc_fraction', 0.5))
    dg_min_epochs = int(_dg.get('min_epochs', 5))
    peak_train_acc = 0.0
    recent_losses: list[float] = []

    for epoch in range(1, cfg.training.epochs + 1):
        train_dataset.set_epoch(epoch)
        t0 = time.time()
        train_loss, train_acc, loss_components = train_epoch(
            pipeline, train_loader, optimizer, ce_fn, supcon_fn,
            loss_w, device, accum_steps, grad_clip)
        final_train_acc = train_acc

        peak_train_acc = max(peak_train_acc, train_acc)
        recent_losses.append(train_loss)
        if len(recent_losses) > dg_window:
            recent_losses.pop(0)
        if dg_on and epoch >= dg_min_epochs and len(recent_losses) == dg_window:
            frozen = (max(recent_losses) - min(recent_losses)) < dg_tol
            if frozen and train_acc < dg_frac * peak_train_acc:
                raise RuntimeError(
                    f"[divergence] training died: train_loss frozen at "
                    f"{train_loss:.4f} for {dg_window} epochs while train_acc "
                    f"fell to {train_acc:.4f} from a peak of {peak_train_acc:.4f} "
                    f"(epoch {epoch}/{cfg.training.epochs}). Refusing to leave a "
                    f"pre-divergence best_model.pt for aggregation. Re-run this "
                    f"cell, or set training.divergence_guard.enabled=false to "
                    f"train through it."
                )
        scheduler.step()

        val_metrics = None
        val_loss, val_acc = None, None
        do_val = (epoch % cfg.training.val_freq == 0) or (epoch == cfg.training.epochs)
        if do_val:
            val_metrics = validate(pipeline, val_loader, ce_fn, device)
            val_loss = val_metrics['loss']
            val_acc = val_metrics['accuracy']

            selection_acc = val_metrics.get('video_accuracy', val_acc)
            if selection_acc > best_val_acc + es_min_delta:
                best_val_acc = selection_acc
                epochs_since_improve = 0
                torch.save(pipeline.state_dict(), ckpt_dir / "best_model.pt")
                if log_model and use_wandb:
                    save_checkpoint(ckpt_dir / "best_model.pt")
            else:
                epochs_since_improve += 1

            if device.type == "cuda":
                torch.cuda.empty_cache()

        lr = scheduler.get_last_lr()[0]
        logger.log_epoch(
            epoch, cfg.training.epochs, train_loss, train_acc,
            val_loss, val_acc, lr=lr, epoch_time=time.time() - t0,
            train_loss_components=loss_components,
            val_metrics=val_metrics,
        )

        if cfg.training.save_every > 0 and epoch % cfg.training.save_every == 0:
            torch.save(pipeline.state_dict(), ckpt_dir / f"epoch_{epoch}.pt")

        if es_on and epoch >= es_min_epochs and epochs_since_improve >= es_patience:
            print(f"[early-stop] no val_acc improvement for {epochs_since_improve} "
                  f"epochs (best={best_val_acc:.4f}); stopping at epoch "
                  f"{epoch}/{cfg.training.epochs}")
            break

    torch.save(pipeline.state_dict(), ckpt_dir / "final_model.pt")
    if log_model and use_wandb:
        save_checkpoint(ckpt_dir / "final_model.pt")
    logger.log_training_summary(final_train_acc, best_val_acc)
    finish_wandb()


if __name__ == "__main__":
    main()
