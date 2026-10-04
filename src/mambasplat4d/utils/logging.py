"""W&B and console logging: Hydra config, epoch metrics, LR, GPU memory, checkpoints."""
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import time

import torch
from omegaconf import DictConfig, OmegaConf

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def init_wandb(cfg: DictConfig) -> bool:
    """wandb.init with resolved Hydra config; False if disabled/unavailable."""
    if not WANDB_AVAILABLE:
        return False

    wcfg = cfg.wandb
    if not wcfg.get('enabled', False) or wcfg.get('mode', 'disabled') == 'disabled':
        return False

    resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)

    tags = list(wcfg.get('tags', []))
    project = wcfg.get('project', 'mambasplat4d')
    feature_mode = cfg.get('dataset', {}).get('feature_mode', '')
    if feature_mode:
        tags.append(feature_mode)
    seed = cfg.get('seed', None)
    if seed is not None:
        tags.append(f'seed{seed}')

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{feature_mode}_{timestamp}" if feature_mode else timestamp

    train_mode = cfg.get('rotation', {}).get('train', None)
    if train_mode is not None:
        tags.append(f'rot_{train_mode}')
    spatial_name = cfg.get('spatial', {}).get('name', None)
    if spatial_name is not None:
        tags.append(spatial_name)
    temporal_name = cfg.get('temporal', {}).get('name', None)
    if temporal_name is not None:
        tags.append(temporal_name)

    wandb.init(
        entity=wcfg.get('entity'),
        project=project,
        name=run_name,
        tags=tags,
        group=wcfg.get('group'),
        notes=wcfg.get('notes'),
        mode=wcfg.get('mode', 'online'),
        config=resolved,
    )

    return True


def watch_model(model, cfg: DictConfig):
    """wandb.watch if cfg.wandb.watch_model."""
    if not WANDB_AVAILABLE or wandb.run is None:
        return
    wcfg = cfg.wandb
    if wcfg.get('watch_model', False):
        wandb.watch(model, log='all', log_freq=wcfg.get('watch_freq', 100))


def save_checkpoint(path: Path):
    """wandb.save checkpoint."""
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.save(str(path), policy='now')


def log_wandb(metrics: Dict[str, Any], step: Optional[int] = None):
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log(metrics, step=step)


def log_gpu_stats() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    stats = {
        'gpu/memory_allocated_gb': torch.cuda.memory_allocated() / 1e9,
        'gpu/memory_reserved_gb': torch.cuda.memory_reserved() / 1e9,
        'gpu/max_memory_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
    }
    log_wandb(stats)
    return stats


def log_summary(key: str, value: Any):
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.run.summary[key] = value


def finish_wandb():
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()


class TrainingLogger:
    """Console + W&B epoch logger."""

    def __init__(self, use_wandb: bool = False, prefix: str = "train"):
        self.use_wandb = use_wandb
        self.prefix = prefix
        self.epoch_times = []
        self.best_val_acc = 0.0
        self.start_time = time.time()

    def log_epoch(
        self,
        epoch: int,
        epochs: int,
        train_loss: float,
        train_acc: float,
        val_loss: Optional[float] = None,
        val_acc: Optional[float] = None,
        lr: float = 0.0,
        epoch_time: float = 0.0,
        train_loss_components: Optional[Dict[str, float]] = None,
        val_metrics: Optional[Dict[str, Any]] = None,
    ):
        """train_loss_components: e.g. ce_loss, aux_ce_loss. val_metrics: accuracy,
        video_accuracy, macro_f1, intra/inter_class_sim, sim_gap, centroid_similarity."""
        self.epoch_times.append(epoch_time)

        line = f"[{self.prefix}] Epoch {epoch}/{epochs}"
        line += f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
        if val_loss is not None:
            line += f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        if val_metrics and 'video_accuracy' in val_metrics:
            line += f"  video_acc={val_metrics['video_accuracy']:.4f}"
        if val_metrics and 'macro_f1' in val_metrics:
            line += f"  macro_f1={val_metrics['macro_f1']:.4f}"
        if val_metrics and 'video_macro_f1' in val_metrics:
            line += f"  video_f1={val_metrics['video_macro_f1']:.4f}"
        if val_metrics and 'sim_gap' in val_metrics:
            line += f"  sim_gap={val_metrics['sim_gap']:.4f}"
        line += f"  lr={lr:.2e}  [{epoch_time:.1f}s]"

        selection_acc = val_metrics.get('video_accuracy', val_acc) if val_metrics else val_acc
        if selection_acc is not None and selection_acc > self.best_val_acc:
            self.best_val_acc = selection_acc
            line += "  *best*"

        print(line, flush=True)

        if not self.use_wandb:
            return

        metrics = {
            'epoch': epoch,
            'run_kind': self.prefix,
            'train/loss': train_loss,
            'train/acc': train_acc,
            'lr': lr,
            'time/epoch_seconds': epoch_time,
        }

        if train_loss_components:
            for key, value in train_loss_components.items():
                metrics[f'train/{key}'] = value

        if val_loss is not None:
            metrics['val/loss'] = val_loss
            metrics['val/acc'] = val_acc
        if val_metrics:
            if 'video_accuracy' in val_metrics:
                metrics['val/video_acc'] = val_metrics['video_accuracy']
            if 'video_macro_f1' in val_metrics:
                metrics['val/video_macro_f1'] = val_metrics['video_macro_f1']
            for key in ('macro_precision', 'macro_recall', 'macro_f1', 'macro_auc_roc'):
                if key in val_metrics:
                    metrics[f'val/{key}'] = val_metrics[key]
            for key in ('intra_class_sim', 'inter_class_sim', 'sim_gap',
                        'centroid_similarity'):
                if key in val_metrics:
                    metrics[f'embedding/{key}'] = val_metrics[key]

        log_wandb(metrics)
        log_gpu_stats()

    def log_training_summary(self, final_train_acc: float, best_val_acc: float):
        total_time = time.time() - self.start_time
        avg_epoch = sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0

        print(f"\n{'='*60}")
        print(f"{self.prefix} training complete")
        print(f"  Total time:     {total_time:.0f}s ({total_time/60:.1f} min)")
        print(f"  Avg epoch:      {avg_epoch:.1f}s")
        print(f"  Final train acc: {final_train_acc:.4f}")
        print(f"  Best val acc:    {best_val_acc:.4f}  (video-level when available)")

        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"  Peak GPU memory: {peak_mem:.2f} GB")

        print(f"{'='*60}", flush=True)

        if self.use_wandb:
            log_summary('run_kind', self.prefix)
            log_summary('best_val_acc', best_val_acc)
            log_summary('final_train_acc', final_train_acc)
            log_summary('total_time_minutes', total_time / 60)
            if torch.cuda.is_available():
                log_summary('peak_gpu_memory_gb', torch.cuda.max_memory_allocated() / 1e9)
