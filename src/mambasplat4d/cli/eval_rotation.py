"""Rotation-robustness eval for MambaSplat-4D on AeroSplat-4D.

Test-time rotation after collation on (B, T, N, 3): one R per clip (global modes),
one R per frame (`so3_per_frame`).

Protocols, train/test: `z/z` (mode `z`, z-trained), `z/SO(3)` (`so3`),
`z/SO(3)`-per-frame (`so3_per_frame`), `SO(3)/SO(3)` (`so3`, so3-trained).
`none` = unrotated.

Rotations from shared Haar cache via `mambasplat4d.rotations`: identical matrices for
every model. Miss generates; mismatch fatal unless `rotation_eval.allow_inline=true`.

Streaming: spatial encoder per frame, one sequence classification.
`rotation_eval.eval_mode=batched`: single fused (B*T) forward.

Video level: softmax summed over a sequence's clips, then argmax.

Usage:
    python -m mambasplat4d.cli.eval_rotation dataset=aerosplat4d_pre \
        training=aerosplat4d spatial=vn_3dgs_medium temporal=vn_mamba \
        +checkpoint=results/aero4d/mambasplat_X/z/seed_42/best_model.pt \
        ++rotation_eval.splits=[test] ++rotation_eval.n_trials=3 \
        ++rotation_eval.seed=42 ++rotation_eval.rotations_dir=rotation_cache \
        ++rotation_eval.output=results/eval_rotation_test.json
"""

import json
import math
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from mambasplat4d import paths, rotations
from mambasplat4d.cli._pipeline import get_device, normalize_checkpoint_keys
from mambasplat4d.datasets.collate import collate_sequences
from mambasplat4d.datasets.isaacsim_pre import IsaacSimPreDataset
from mambasplat4d.metrics.classification import compute_extended_metrics
from mambasplat4d.models.pipeline import ClassificationPipeline
from mambasplat4d.utils.feature_config import FEATURE_MODES
from mambasplat4d.utils.pytorch3d_transforms import random_rotations
from mambasplat4d.utils.seed import set_seed


EVAL_MODES = ["none", "z", "so3", "so3_per_frame"]


# Inline sampling fallback (no cache). Same Haar SO(3) sampler as
# mambasplat4d.cli.generate_rotations (vendored PyTorch3D).


def random_rotation_matrix_z_batch(batch_size, device, dtype=torch.float32):
    """Random Z rotations per sequence, (B, 3, 3). Uniform [0, 2π) is Haar on SO(2)."""
    angles = torch.rand(batch_size, device=device, dtype=dtype) * 2 * math.pi
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    R = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    R[:, 0, 0] = cos_a
    R[:, 0, 1] = -sin_a
    R[:, 1, 0] = sin_a
    R[:, 1, 1] = cos_a
    return R


def random_rotation_matrix_so3_batch(batch_size, device, dtype=torch.float32):
    """Haar SO(3) rotations per sequence, (B, 3, 3). Wraps vendored `pytorch3d.random_rotations`."""
    return random_rotations(batch_size, dtype=dtype, device=device)


def _rotation_matrix_to_quaternion(R):
    """3x3 rotation matrix -> [w, x, y, z] quaternion."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / torch.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = torch.stack([w, x, y, z])
    return q / q.norm()


def _rotate_quaternions(q, R):
    """Rotate quaternions q (..., 4) [w,x,y,z] by R (3,3)."""
    q_R = _rotation_matrix_to_quaternion(R)
    w1, x1, y1, z1 = q_R[0], q_R[1], q_R[2], q_R[3]
    w2, x2, y2, z2 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    result = torch.stack([w, x, y, z], dim=-1)
    return F.normalize(result, dim=-1)


def _rotate_quaternions_batch(q, R_batch):
    """Rotate q (B, ..., 4) by R_batch (B, 3, 3); same shape as q."""
    return torch.stack(
        [_rotate_quaternions(q[i], R_batch[i]) for i in range(q.shape[0])], dim=0
    )


def build_test_dataset(cfg, split="test"):
    """Eval dataset for split, no augmentation. Returns (dataset, feature_keys)."""
    ds_cfg = cfg.dataset
    feature_mode = ds_cfg.get("feature_mode", "X")
    feature_keys = FEATURE_MODES[feature_mode]

    fmt = ds_cfg.format
    if fmt != "isaacsim_pre":
        raise ValueError(
            f"eval_rotation: unsupported dataset format '{fmt}'; only 'isaacsim_pre' "
            f"(AeroSplat-4D preprocessed) is supported"
        )

    dataset = IsaacSimPreDataset(
        data_root=ds_cfg.data_root,
        split=split,
        num_points=ds_cfg.num_points,
        seq_len=ds_cfg.get("seq_len", None),
        feature_keys=feature_keys,
        frame_interval=ds_cfg.get("frame_interval", 1),
        n_versions=ds_cfg.get("n_versions", 25),
    )
    return dataset, feature_keys


def _encode_frame(pipeline, frame_data, mask_t):
    """Spatial encoder + VN-In bridge on one frame; returns (B, d_mamba).

    Temporal encoder runs with T=1; only bridge output kept. Sequence pass happens
    in `_classify_sequence`.
    """
    V_t, X_inv_t = pipeline.spatial_encoder(frame_data, mask=mask_t)
    _, h_t = pipeline.temporal_encoder(
        V_t,
        mask=mask_t,
        T=1,
        X_inv=X_inv_t,
        gaussian_data=frame_data,
    )
    return h_t


def _classify_sequence(pipeline, frame_features):
    """Pos-enc + Mamba + pooling + classifier; mirrors `VNMambaTemporalEncoder.forward` post-bridge."""
    if getattr(pipeline, "per_frame_mode", False):
        logits = torch.stack([pipeline.frame_classifier(h) for h in frame_features])
        return logits.mean(dim=0)

    seq = torch.stack(frame_features, dim=1)  # (B, T, d_mamba)
    h_seq = pipeline.temporal_encoder.pos_encoder(seq)
    h_seq = pipeline.temporal_encoder.mamba(h_seq)
    pooling = getattr(pipeline.temporal_encoder, "pooling", "mean")
    if pooling == "last":
        features = h_seq[:, -1]
    else:
        features = h_seq.mean(dim=1)
    return pipeline.classifier(features)


# Batched eval: flatten (B, T, N, F) to (B*T, N, F), one pipeline(..., T=T) forward.
# Same math as streaming for vn_mamba without per-frame heads.
# Opt-in: +rotation_eval.eval_mode=batched.

_BATCHED_EVAL_FALLBACK_WARNED = False


def _batched_eval_supported(pipeline):
    """(supported, reason); reason '' iff supported."""
    if getattr(pipeline, "per_frame_mode", False):
        return (
            False,
            "pipeline.per_frame_mode=True (forward returns frame-level outputs)",
        )
    return True, ""


def _warn_batched_fallback(reason):
    global _BATCHED_EVAL_FALLBACK_WARNED
    if not _BATCHED_EVAL_FALLBACK_WARNED:
        print(
            f"WARNING: eval_mode='batched' requested but {reason}; falling back to streaming."
        )
        _BATCHED_EVAL_FALLBACK_WARNED = True


def _batched_clip_forward(
    pipeline, batch, feature_keys, R_seq, R_frame, T, B, N, device
):
    """Fused (B*T) forward, as train.py to_device + pipeline.forward. Logits (B, num_classes)."""
    gaussian_data = {k: batch[k].to(device) for k in feature_keys if k in batch}
    mask = batch["mask"].to(device)  # (B, T, N)

    R_full = None
    if R_seq is not None:
        R_full = R_seq.unsqueeze(1).expand(B, T, 3, 3)  # (B, T, 3, 3)
    elif R_frame is not None:
        R_full = R_frame  # (B, T, 3, 3)

    if R_full is not None:
        gaussian_data["position"] = torch.einsum(
            "btnd,btdc->btnc",
            gaussian_data["position"],
            R_full.transpose(-1, -2),
        )
        if "quaternion" in gaussian_data:
            q = gaussian_data["quaternion"]  # (B, T, N, 4)
            q_flat = q.reshape(B * T, *q.shape[2:])
            R_flat = R_full.reshape(B * T, 3, 3)
            q_rot_flat = _rotate_quaternions_batch(q_flat, R_flat)
            gaussian_data["quaternion"] = q_rot_flat.reshape(q.shape)

    gaussian_data = {k: v.reshape(B * T, N, -1) for k, v in gaussian_data.items()}
    mask_flat = mask.reshape(B * T, N)

    out = pipeline(gaussian_data, mask=mask_flat, T=T)
    return out["logits"]


def eval_trial(
    pipeline,
    test_loader,
    feature_keys,
    mode,
    device,
    trial_seed,
    num_classes,
    rotation_matrices=None,
    eval_mode="streaming",
):
    """One eval trial with optional rotation.

    rotation_matrices: per-clip cache, (n_clips, 3, 3) or (n_clips, T, 3, 3); replaces inline sampling.
    Returns dict: video_acc, video_class_acc, clip_acc, clip_class_acc, plus raw
    clip_probs / clip_labels / video_probs / video_labels for extended metrics.
    """
    torch.manual_seed(trial_seed)
    np.random.seed(trial_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(trial_seed)

    agg_prob = {}
    agg_label = {}
    sample_counter = 0
    clip_offset = 0
    clip_preds = []
    clip_probs_chunks = []
    clip_labels_chunks = []

    with torch.no_grad():
        for batch in tqdm(
            test_loader,
            desc=f"{mode} seed={trial_seed}",
            dynamic_ncols=True,
        ):
            labels = batch["label"].to(device)
            B, T, N = batch["position"].shape[:3]

            R_seq = None
            R_frame = None
            if rotation_matrices is not None and mode != "none":
                remaining = rotation_matrices.shape[0] - clip_offset
                if remaining < B:
                    raise RuntimeError(
                        f"rotation cache exhausted at clip {clip_offset}: "
                        f"requested B={B} but only {remaining} remain "
                        f"(cache len {rotation_matrices.shape[0]}, "
                        f"dataset len {len(test_loader.dataset)}). "
                        f"Mode={mode}, cache shape={tuple(rotation_matrices.shape)}. "
                        f"Regenerate via mambasplat4d.cli.generate_rotations or check "
                        f"that dataset.seq_len/frame_interval match "
                        f"rotation_eval.seq_len/frame_interval."
                    )
                R = rotation_matrices[clip_offset : clip_offset + B].to(
                    device=device, dtype=torch.float32
                )
                if R.ndim == 3:
                    R_seq = R
                else:
                    R_frame = R
            elif mode == "z":
                R_seq = random_rotation_matrix_z_batch(B, device=device)
            elif mode == "so3":
                R_seq = random_rotation_matrix_so3_batch(B, device=device)
            elif mode == "so3_per_frame":
                R_frame = random_rotation_matrix_so3_batch(B * T, device=device).view(
                    B, T, 3, 3
                )
            elif mode != "none":
                raise ValueError(f"Unknown rotation mode: {mode!r}")
            clip_offset += B

            use_batched = False
            if eval_mode == "batched":
                ok, reason = _batched_eval_supported(pipeline)
                if ok:
                    use_batched = True
                else:
                    _warn_batched_fallback(reason)

            if use_batched:
                logits = _batched_clip_forward(
                    pipeline,
                    batch,
                    feature_keys,
                    R_seq,
                    R_frame,
                    T=T,
                    B=B,
                    N=N,
                    device=device,
                )
            else:
                frame_features = []
                for t in range(T):
                    pos_t = batch["position"][:, t].to(device)  # (B, N, 3)
                    mask_t = batch["mask"][:, t].to(device)  # (B, N)

                    frame_data = {"position": pos_t}
                    for k in feature_keys:
                        if k != "position" and k in batch:
                            frame_data[k] = batch[k][:, t].to(device)

                    R_t = None
                    if R_seq is not None:
                        R_t = R_seq
                    elif R_frame is not None:
                        R_t = R_frame[:, t]

                    if R_t is not None:
                        frame_data["position"] = torch.einsum(
                            "bnd,bdc->bnc", pos_t, R_t.transpose(1, 2)
                        )
                        if "quaternion" in frame_data:
                            frame_data["quaternion"] = _rotate_quaternions_batch(
                                frame_data["quaternion"], R_t
                            )

                    frame_features.append(_encode_frame(pipeline, frame_data, mask_t))

                logits = _classify_sequence(pipeline, frame_features)

            prob = F.softmax(logits, dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()

            clip_probs_chunks.append(prob)
            clip_labels_chunks.append(labels_np.copy())

            video_idx_np = (
                batch["video_idx"].cpu().numpy() if "video_idx" in batch else None
            )

            for i in range(B):
                clip_pred = int(np.argmax(prob[i]))
                clip_true = int(labels_np[i])
                clip_preds.append((clip_pred, clip_true))

                if video_idx_np is not None:
                    vid = int(video_idx_np[i])
                else:
                    vid = sample_counter
                    sample_counter += 1

                if vid in agg_prob:
                    agg_prob[vid] += prob[i]
                else:
                    agg_prob[vid] = prob[i].copy()
                    agg_label[vid] = int(labels_np[i])

    clip_correct = [p == t for p, t in clip_preds]
    clip_accuracy = float(np.mean(clip_correct)) if clip_correct else 0.0

    clip_class_count = [0] * num_classes
    clip_class_correct = [0] * num_classes
    for p, t in clip_preds:
        clip_class_count[t] += 1
        clip_class_correct[t] += p == t
    clip_class_acc = [
        c / float(s) if s > 0 else 0.0
        for c, s in zip(clip_class_correct, clip_class_count)
    ]

    agg_pred = {k: int(np.argmax(v)) for k, v in agg_prob.items()}
    pred_correct = [agg_pred[k] == agg_label[k] for k in agg_pred]
    accuracy = float(np.mean(pred_correct))

    class_count = [0] * num_classes
    class_correct = [0] * num_classes
    for k, v in agg_pred.items():
        lbl = agg_label[k]
        class_count[lbl] += 1
        class_correct[lbl] += v == lbl
    class_acc = [
        c / float(s) if s > 0 else 0.0 for c, s in zip(class_correct, class_count)
    ]

    if clip_probs_chunks:
        clip_probs_arr = np.concatenate(clip_probs_chunks, axis=0)
        clip_labels_arr = np.concatenate(clip_labels_chunks, axis=0)
    else:
        clip_probs_arr = np.zeros((0, num_classes), dtype=np.float32)
        clip_labels_arr = np.zeros((0,), dtype=np.int64)

    # Video level: renormalise summed softmax (mean over clips); sort vids for determinism.
    if agg_prob:
        sorted_vids = sorted(agg_prob.keys())
        video_probs_arr = np.stack(
            [agg_prob[v] / max(agg_prob[v].sum(), 1e-12) for v in sorted_vids],
            axis=0,
        )
        video_labels_arr = np.asarray(
            [agg_label[v] for v in sorted_vids],
            dtype=np.int64,
        )
    else:
        video_probs_arr = np.zeros((0, num_classes), dtype=np.float32)
        video_labels_arr = np.zeros((0,), dtype=np.int64)

    return {
        "video_acc": accuracy,
        "video_class_acc": class_acc,
        "clip_acc": clip_accuracy,
        "clip_class_acc": clip_class_acc,
        "clip_probs": clip_probs_arr,
        "clip_labels": clip_labels_arr,
        "video_probs": video_probs_arr,
        "video_labels": video_labels_arr,
    }


def _fail_or_warn_inline(reason, allow_inline):
    message = (
        f"{reason}. Inline rotation sampling would make this run incomparable to "
        f"every other row of its table (different rotations per model). Generate the "
        f"cache with `python -m {rotations.GENERATOR_MODULE}` or pass "
        f"rotation_eval.allow_inline=true to override."
    )
    if not allow_inline:
        raise RuntimeError(message)
    print(f"WARNING: {message}")


def _load_rotation_cache(
    cfg, split, dataset, seed, n_trials, rotations_dir, allow_inline
):
    """Resolve, generate if missing, verify cache for split. None if unresolved and inline allowed."""
    if not rotations_dir:
        _fail_or_warn_inline("rotation_eval.rotations_dir is unset", allow_inline)
        return None
    try:
        rot_path = rotations.ensure(
            cfg,
            split,
            n_trials=n_trials,
            seed=seed,
            rotations_dir=rotations_dir,
            data_root=cfg.dataset.data_root,
            n_clips=len(dataset),
        )
    except rotations.RotationCacheError as exc:
        _fail_or_warn_inline(str(exc), allow_inline)
        return None
    rot_data = torch.load(str(rot_path), map_location="cpu", weights_only=True)
    print(f"Loaded pre-generated rotations from {rot_path}")
    return rot_data


@hydra.main(version_base=None, config_path=paths.CONFIG_DIR, config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    device = get_device(cfg)

    rot_eval = cfg.get("rotation_eval", {})
    n_trials = int(rot_eval.get("n_trials", 5))
    seed = int(rot_eval.get("seed", cfg.seed))
    model_name = str(
        rot_eval.get(
            "model_name",
            f"{cfg.spatial.name} + {cfg.temporal.name}",
        )
    )
    allow_inline = bool(rot_eval.get("allow_inline", False))

    eval_splits_raw = rot_eval.get("splits", "test")
    if isinstance(eval_splits_raw, str):
        eval_splits = [s.strip() for s in eval_splits_raw.split(",") if s.strip()]
    else:
        eval_splits = list(eval_splits_raw)

    output_path = rot_eval.get("output", None)
    if output_path is None:
        output_path = (
            Path(HydraConfig.get().runtime.output_dir)
            / f"eval_rotation_{'_'.join(eval_splits)}.json"
        )
    output_path = Path(str(output_path))

    # Cache file: rotation_matrices_{split}_seed{seed}_T{seq_len}_dt{frame_interval}_n{n_trials}.pt
    rotations_dir = str(rot_eval.get("rotations_dir", ""))
    rot_seq_len = int(rot_eval.get("seq_len", cfg.dataset.get("seq_len", 12)))
    rot_frame_interval = int(
        rot_eval.get("frame_interval", cfg.dataset.get("frame_interval", 1))
    )

    eval_mode = str(rot_eval.get("eval_mode", "streaming"))
    if eval_mode not in ("streaming", "batched"):
        raise ValueError(
            f"rotation_eval.eval_mode must be 'streaming' or 'batched', got {eval_mode!r}"
        )
    print(f"Eval mode: {eval_mode}")

    modes_raw = rot_eval.get("modes", None)
    if modes_raw is None:
        eval_modes_list = list(EVAL_MODES)
    else:
        eval_modes_list = [str(m) for m in modes_raw]
        unknown = [m for m in eval_modes_list if m not in EVAL_MODES]
        if unknown:
            raise ValueError(
                f"rotation_eval.modes contains unknown entries {unknown}; "
                f"valid: {EVAL_MODES}"
            )
    print(f"Rotation modes: {eval_modes_list}")

    # Cache name encodes (T, dt); dataset must use same stride/span or per-clip
    # rotation index overruns.
    prev_struct = OmegaConf.is_struct(cfg)
    OmegaConf.set_struct(cfg, False)
    try:
        ds_seq_len = cfg.dataset.get("seq_len", None)
        ds_frame_interval = cfg.dataset.get("frame_interval", None)
        if ds_seq_len is None or int(ds_seq_len) != rot_seq_len:
            print(
                f"[eval_rotation] dataset.seq_len {ds_seq_len} -> {rot_seq_len} "
                f"(from rotation_eval.seq_len)"
            )
        cfg.dataset.seq_len = rot_seq_len
        if ds_frame_interval is None or int(ds_frame_interval) != rot_frame_interval:
            print(
                f"[eval_rotation] dataset.frame_interval {ds_frame_interval} -> "
                f"{rot_frame_interval} (from rotation_eval.frame_interval)"
            )
        cfg.dataset.frame_interval = rot_frame_interval
    finally:
        OmegaConf.set_struct(cfg, prev_struct)

    if hasattr(cfg.dataset, "num_classes"):
        cfg.num_classes = cfg.dataset.num_classes
    num_classes = cfg.num_classes

    ckpt_path = cfg.get("checkpoint", None)

    # num_classes from checkpoint avoids size mismatch
    if ckpt_path:
        state_dict_raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state_dict = normalize_checkpoint_keys(state_dict_raw)
        for key in ("classifier.classifier.3.weight", "classifier.classifier.3.bias"):
            if key in state_dict:
                ckpt_num_classes = state_dict[key].shape[0]
                if ckpt_num_classes != num_classes:
                    print(
                        f"WARNING: checkpoint has {ckpt_num_classes} classes, "
                        f"config has {num_classes}; overriding to match checkpoint"
                    )
                    cfg.num_classes = ckpt_num_classes
                    num_classes = ckpt_num_classes
                break

    pipeline = ClassificationPipeline(cfg).to(device)
    if not pipeline.vn_mamba_mode:
        raise ValueError(
            f"eval_rotation: unsupported temporal encoder '{cfg.temporal.name}'; "
            f"only 'vn_mamba' is supported"
        )

    if ckpt_path:
        model_state = pipeline.state_dict()
        filtered = {}
        skipped = []
        for k, v in state_dict.items():
            if k not in model_state:
                continue
            if model_state[k].shape != v.shape:
                skipped.append(
                    f"  {k}: ckpt {list(v.shape)} vs model {list(model_state[k].shape)}"
                )
            else:
                filtered[k] = v
        if skipped:
            print(f"Skipped {len(skipped)} size-mismatched keys:")
            for s in skipped:
                print(s)
        missing, unexpected = pipeline.load_state_dict(filtered, strict=False)
        if missing:
            print(f"WARNING: Missing {len(missing)} model keys after load.")
        if unexpected:
            print(f"WARNING: Unexpected {len(unexpected)} checkpoint keys after load.")
        print(
            f"Loaded {len(filtered)} / {len(model_state)} model tensors from checkpoint."
        )
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("WARNING: No checkpoint specified, evaluating random weights")

    pipeline.eval()

    all_split_results = {}

    for split in eval_splits:
        print(f"\n{'=' * 60}")
        print(f"  Evaluating split: {split}")
        print(f"{'=' * 60}")

        split_dataset, feature_keys = build_test_dataset(cfg, split=split)
        print(f"Feature mode keys: {feature_keys}")
        print(
            f"{split} clips: {len(split_dataset)}, sequences: {len(split_dataset.sequences)}"
        )
        nw = cfg.dataset.num_workers
        split_loader = DataLoader(
            split_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=nw,
            collate_fn=collate_sequences,
            pin_memory=True,
            persistent_workers=nw > 0,
            multiprocessing_context="spawn" if nw > 0 else None,
        )

        print(f"Evaluating {model_name} on {len(split_dataset)} {split} clips")

        rot_data = _load_rotation_cache(
            cfg,
            split,
            split_dataset,
            seed,
            n_trials,
            rotations_dir,
            allow_inline,
        )

        split_results = {}
        for mode in eval_modes_list:
            n = 1 if mode == "none" else n_trials
            trial_video_accs = []
            trial_video_class_accs = []
            trial_clip_accs = []
            trial_clip_class_accs = []
            trial_clip_probs = []
            trial_clip_labels = []
            trial_video_probs = []
            trial_video_labels = []

            for trial in range(n):
                trial_seed = seed + trial

                rot_mats = None
                if rot_data is not None and mode != "none":
                    key = f"{mode}_{trial}"
                    rot_mats = rot_data.get(key)
                    if rot_mats is not None:
                        print(
                            f"  Using pre-generated rotations: {key} shape={tuple(rot_mats.shape)}"
                        )
                    else:
                        _fail_or_warn_inline(
                            f"key '{key}' not in rotation cache",
                            allow_inline,
                        )

                print(f"\n[{split}/{mode}] Trial {trial + 1}/{n} (seed={trial_seed})")

                trial_result = eval_trial(
                    pipeline,
                    split_loader,
                    feature_keys,
                    mode,
                    device,
                    trial_seed,
                    num_classes,
                    rotation_matrices=rot_mats,
                    eval_mode=eval_mode,
                )
                trial_video_accs.append(trial_result["video_acc"])
                trial_video_class_accs.append(trial_result["video_class_acc"])
                trial_clip_accs.append(trial_result["clip_acc"])
                trial_clip_class_accs.append(trial_result["clip_class_acc"])
                trial_clip_probs.append(trial_result["clip_probs"])
                trial_clip_labels.append(trial_result["clip_labels"])
                trial_video_probs.append(trial_result["video_probs"])
                trial_video_labels.append(trial_result["video_labels"])
                print(f"  Video Acc@1: {trial_result['video_acc']:.4f}")
                print(f"  Clip  Acc@1: {trial_result['clip_acc']:.4f}")

            # Per-trial extended metrics, for std across trials.
            ext_clip_per_trial = [
                compute_extended_metrics(p, lab, num_classes)
                for p, lab in zip(trial_clip_probs, trial_clip_labels)
            ]
            ext_video_per_trial = [
                compute_extended_metrics(p, lab, num_classes)
                for p, lab in zip(trial_video_probs, trial_video_labels)
            ]

            def _agg(metric_dicts, key):
                vals = [float(d[key]) for d in metric_dicts]
                return float(np.mean(vals)) if vals else 0.0, float(
                    np.std(vals)
                ) if vals else 0.0

            mode_summary = {
                "mean": float(np.mean(trial_video_accs)),
                "std": float(np.std(trial_video_accs)),
                "trials": trial_video_accs,
                "class_acc": trial_video_class_accs,
                "clip_mean": float(np.mean(trial_clip_accs)),
                "clip_std": float(np.std(trial_clip_accs)),
                "clip_trials": trial_clip_accs,
                "clip_class_acc": trial_clip_class_accs,
            }

            for tag, dicts in (
                ("clip", ext_clip_per_trial),
                ("video", ext_video_per_trial),
            ):
                for k in (
                    "macro_f1",
                    "cohen_kappa",
                    "mcc",
                    "nll",
                    "brier",
                    "macro_auc_roc",
                ):
                    mu, sd = _agg(dicts, k)
                    mode_summary[f"{tag}_{k}"] = mu
                    mode_summary[f"{tag}_{k}_std"] = sd

            # Confusion matrices over concatenated trials.
            cat_clip_probs = (
                np.concatenate(trial_clip_probs, axis=0)
                if trial_clip_probs
                else np.zeros((0, num_classes))
            )
            cat_clip_labels = (
                np.concatenate(trial_clip_labels, axis=0)
                if trial_clip_labels
                else np.zeros((0,), dtype=np.int64)
            )
            cat_video_probs = (
                np.concatenate(trial_video_probs, axis=0)
                if trial_video_probs
                else np.zeros((0, num_classes))
            )
            cat_video_labels = (
                np.concatenate(trial_video_labels, axis=0)
                if trial_video_labels
                else np.zeros((0,), dtype=np.int64)
            )

            ext_clip_cat = compute_extended_metrics(
                cat_clip_probs,
                cat_clip_labels,
                num_classes,
            )
            ext_video_cat = compute_extended_metrics(
                cat_video_probs,
                cat_video_labels,
                num_classes,
            )

            mode_summary["clip_confusion_matrix"] = ext_clip_cat["confusion_matrix"]
            mode_summary["video_confusion_matrix"] = ext_video_cat["confusion_matrix"]

            split_results[mode] = mode_summary

        all_split_results[split] = split_results

    output_data = {
        "model": model_name,
        "dataset": cfg.dataset.name,
        "checkpoint": str(ckpt_path) if ckpt_path else None,
        "seed": seed,
        "n_trials": n_trials,
        "results": all_split_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")

    for split in eval_splits:
        print(f"\n{'=' * 60}")
        print(f"  {model_name}: {split} split")
        print(f"{'=' * 60}")
        for mode in eval_modes_list:
            r = all_split_results[split][mode]
            vid_str = f"{r['mean'] * 100:.1f}%"
            if r["std"] > 0:
                vid_str += f" +/- {r['std'] * 100:.1f}%"
            clip_str = f"{r['clip_mean'] * 100:.1f}%"
            if r["clip_std"] > 0:
                clip_str += f" +/- {r['clip_std'] * 100:.1f}%"
            print(f"  {mode:>13s}:  clip={clip_str:<20s}  video={vid_str}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
