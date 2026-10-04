#!/usr/bin/env python3
"""LGM batch reconstruction: four static views per frame to one PLY (or torch dict).

Layout <output>/<class>/<sequence>/frame_XXXX.ply, consumed by
mambasplat4d.preprocessing.preprocess_aerosplat4d. Needs LGM checkout
(https://github.com/3DTopia/LGM) with pipelines/lgm/lgm.patch applied, via LGM_ROOT or --lgm-root.

    LGM_ROOT=/path/to/LGM python pipelines/lgm/batch_reconstruct.py \
        --input  $AEROSPLAT_RENDER_ROOT/aerosplat4d_renders \
        --output $AEROSPLAT_RENDER_ROOT/aerosplat4d_lgm_ply \
        --checkpoint /path/to/LGM/pretrained/model_fp16_fixrot.safetensors
"""

import os
import re
import sys
import json
import time
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file


def _lgm_root() -> str:
    for i, a in enumerate(sys.argv):
        if a == "--lgm-root" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--lgm-root="):
            return a.split("=", 1)[1]
    root = os.environ.get("LGM_ROOT")
    if not root:
        sys.exit("[ERROR] set LGM_ROOT (or pass --lgm-root) to a 3DTopia/LGM checkout with lgm.patch applied")
    return root


LGM_ROOT = os.path.abspath(_lgm_root())
if LGM_ROOT not in sys.path:
    sys.path.insert(0, LGM_ROOT)
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
if os.path.isdir(SRC_ROOT) and SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from core.options import Options
from core.models import LGM
from mambasplat4d.categories import CATEGORIES, CATEGORY_NAMES

CAMERAS = ["cam_01", "cam_02", "cam_03", "cam_04"]

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def make_big_options() -> Options:
    """'big' Options matching pretrained checkpoint."""
    return Options(
        input_size=256,
        up_channels=(1024, 1024, 512, 256, 128),
        up_attention=(True, True, True, False, False),
        splat_size=128,
        output_size=512,
        batch_size=1,
        num_views=8,
        gradient_accumulation_steps=1,
        mixed_precision="bf16",
        lambda_lpips=0,
    )


def load_image_with_mask(rgb_path: str, mask_path: str):
    """rgb float32 [H, W, 3] in [0, 1]; mask float32 [H, W] in {0, 1}."""
    rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
    mask_raw = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
    mask = (mask_raw > 0.5).astype(np.float32)
    return rgb, mask


def recenter_and_crop(rgb: np.ndarray, mask: np.ndarray,
                      border_ratio: float = 0.2,
                      output_size: int = 256) -> np.ndarray:
    """Square crop around mask, white composite, resize."""
    H, W = mask.shape

    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        side = min(H, W)
        y0, x0 = (H - side) // 2, (W - side) // 2
        crop_rgb = rgb[y0:y0 + side, x0:x0 + side]
        crop_mask = mask[y0:y0 + side, x0:x0 + side]
    else:
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())

        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        half = max(x_max - x_min, y_max - y_min) / 2.0
        half *= 1.0 + border_ratio

        x0, y0 = int(cx - half), int(cy - half)
        x1, y1 = int(cx + half), int(cy + half)

        pad_l = max(0, -x0)
        pad_t = max(0, -y0)
        pad_r = max(0, x1 - W)
        pad_b = max(0, y1 - H)

        if pad_l or pad_t or pad_r or pad_b:
            rgb = np.pad(rgb, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)),
                         mode="constant", constant_values=1.0)
            mask = np.pad(mask, ((pad_t, pad_b), (pad_l, pad_r)),
                          mode="constant", constant_values=0.0)
            x0 += pad_l; y0 += pad_t
            x1 += pad_l; y1 += pad_t

        crop_rgb = rgb[y0:y1, x0:x1]
        crop_mask = mask[y0:y1, x0:x1]

    m = crop_mask[..., np.newaxis]
    white_bg = crop_rgb * m + (1.0 - m)

    pil = Image.fromarray((np.clip(white_bg, 0, 1) * 255).astype(np.uint8))
    pil = pil.resize((output_size, output_size), Image.LANCZOS)
    return np.array(pil, dtype=np.float32) / 255.0


def save_gaussian_pt(gaussians: torch.Tensor, path: str):
    """gaussians [1, N, 14] from forward_gaussians() (pos, opacity, scale, quat, SH DC) to depthsplat .pt dict."""
    g = gaussians[0]

    # same prune threshold as save_ply
    opacity = g[:, 3:4]
    valid = opacity.squeeze(-1) >= 0.005
    g = g[valid]

    save_dict = {
        "position": g[:, 0:3].cpu(),
        "opacity": g[:, 3:4].cpu(),
        "scale": g[:, 4:7].cpu(),
        "quaternion": g[:, 7:11].cpu(),
        "sh_dc": g[:, 11:14].cpu(),
    }
    torch.save(save_dict, path)


@dataclass
class BatchConfig:
    input_dir: str = ""
    output_dir: str = ""
    checkpoint: str = os.path.join(LGM_ROOT, "pretrained",
                                   "model_fp16_fixrot.safetensors")
    frame_start: Optional[int] = None
    frame_end: Optional[int] = None
    output_format: str = "ply"
    skip_existing: bool = True
    # no splitting here; preprocessing uses assets/splits/train-test-val_split.yaml
    device: str = "cuda"
    border_ratio: float = 0.2


@dataclass
class ProcessingStats:
    total_sequences: int = 0
    frames_saved: int = 0
    frames_skipped: int = 0
    frames_failed: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time


class BatchReconstructor:
    def __init__(self, config: BatchConfig):
        self.config = config
        self.model = None
        self.opt = None
        self.rays = None
        self.stats = ProcessingStats()

    def load_model(self):
        """Load LGM once; prepare default rays."""
        print("[MODEL] Loading LGM (big config, FP16) ...")
        self.opt = make_big_options()
        self.model = LGM(self.opt)

        if not os.path.isfile(self.config.checkpoint):
            sys.exit(f"[ERROR] Checkpoint not found: {self.config.checkpoint}")

        ckpt = load_file(self.config.checkpoint, device="cpu")
        self.model.load_state_dict(ckpt, strict=False)

        device = torch.device(self.config.device)
        self.model = self.model.half().to(device)
        self.model.eval()

        self.rays = self.model.prepare_default_rays(device)
        n_gauss = 4 * self.opt.splat_size ** 2
        print(f"[MODEL] Loaded on {device}, "
              f"splat_size={self.opt.splat_size} -> {n_gauss} Gaussians")

    def discover_sequences(self) -> list[dict]:
        """Sequences with .render_complete: [{path, name, category, category_id}]."""
        sequences = []
        input_dir = Path(self.config.input_dir)

        for cat_name, cat_id in CATEGORIES.items():
            cat_dir = input_dir / cat_name
            if not cat_dir.is_dir():
                continue

            for seq_dir in sorted(cat_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue
                if not (seq_dir / ".render_complete").exists():
                    continue

                sequences.append({
                    "path": str(seq_dir),
                    "name": seq_dir.name,
                    "category": cat_name,
                    "category_id": cat_id,
                })

        return sequences

    def get_num_frames(self, seq_path: str) -> int:
        """Frame count from metadata, else cam_01 RGB count."""
        meta_path = os.path.join(seq_path, "drone_camera_observations.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            if isinstance(meta, dict):
                if "metadata" in meta and "num_frames" in meta["metadata"]:
                    return int(meta["metadata"]["num_frames"])
                if "frames" in meta and isinstance(meta["frames"], list):
                    return len(meta["frames"])
            if isinstance(meta, list) and len(meta) > 0:
                return len(meta)

        rgb_dir = os.path.join(seq_path, "cam_01", "rgb")
        if os.path.isdir(rgb_dir):
            return len([f for f in os.listdir(rgb_dir) if f.endswith(".png")])
        return 0

    def _load_views(self, seq_path: str, frame_idx: int
                    ) -> Optional[np.ndarray]:
        """4 cropped views, [4, 256, 256, 3] float32; None if any missing."""
        views = []
        for cam_name in CAMERAS:
            rgb_path = os.path.join(seq_path, cam_name, "rgb",
                                    f"rgb_{frame_idx:04d}.png")
            mask_path = os.path.join(seq_path, cam_name, "mask",
                                     f"drone_mask_{frame_idx:04d}.png")

            if not os.path.isfile(rgb_path) or not os.path.isfile(mask_path):
                return None

            rgb, mask = load_image_with_mask(rgb_path, mask_path)
            img = recenter_and_crop(rgb, mask,
                                    border_ratio=self.config.border_ratio,
                                    output_size=self.opt.input_size)
            views.append(img)

        return np.stack(views, axis=0)

    @staticmethod
    def _get_sequence_cam_params(seq_path: str) -> tuple:
        """(elevation, azimuth_offset) from folder name, e.g. 'blank_lgm_10r_15el_30az_...'. Current rigs are static (0, 0); nonzero = legacy."""
        basename = os.path.basename(seq_path)
        elev_match = re.search(r'_(-?\d+)el', basename)
        az_match = re.search(r'_(-?\d+)az', basename)
        elevation = float(elev_match.group(1)) if elev_match else 0.0
        azimuth_offset = float(az_match.group(1)) if az_match else 0.0
        return elevation, azimuth_offset

    def _preprocess_views(self, views_np: np.ndarray) -> torch.Tensor:
        """Normalised views + rays, [1, 4, 9, H, W] for forward_gaussians()."""
        device = torch.device(self.config.device)
        input_tensor = (torch.from_numpy(views_np)
                        .permute(0, 3, 1, 2).float().to(device))
        input_tensor = F.interpolate(input_tensor,
                                     size=(self.opt.input_size,
                                           self.opt.input_size),
                                     mode="bilinear", align_corners=False)
        input_tensor = TF.normalize(input_tensor,
                                    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        return torch.cat([input_tensor, self.rays], dim=1).unsqueeze(0)

    def process_frame(self, seq_path: str, frame_idx: int
                      ) -> Optional[torch.Tensor]:
        """LGM forward on one frame: [1, N, 14] Gaussians, or None if views missing."""
        views_np = self._load_views(seq_path, frame_idx)
        if views_np is None:
            return None

        input_combined = self._preprocess_views(views_np)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                gaussians = self.model.forward_gaussians(input_combined)

        return gaussians

    def process_sequence(self, seq_info: dict,
                         pbar: Optional[tqdm] = None) -> list[str]:
        """All frames of one sequence. Returns relative output paths."""
        seq_path = seq_info["path"]
        seq_name = seq_info["name"]
        cat_id = seq_info["category_id"]

        # recompute rays when camera params change
        elevation, azimuth_offset = self._get_sequence_cam_params(seq_path)
        cache_key = (elevation, azimuth_offset)
        if not hasattr(self, '_cam_cache_key') or self._cam_cache_key != cache_key:
            device = torch.device(self.config.device)
            # IsaacSim Z-up to kiui Y-up: negate el (+el above vs below), az+90 (+X vs +Z zero)
            kiui_el = -elevation
            kiui_az = azimuth_offset + 90
            self.rays = self.model.prepare_default_rays(
                device, elevation=kiui_el, azimuth_offset=kiui_az,
            )
            self._cam_cache_key = cache_key
            tqdm.write(f"[RAYS] Recomputed: IsaacSim el={elevation} az={azimuth_offset}"
                       f" -> kiui el={kiui_el} az={kiui_az}")

        num_frames = self.get_num_frames(seq_path)
        if num_frames == 0:
            tqdm.write(f"[WARN] No frames found in {seq_name}")
            return []

        start = self.config.frame_start if self.config.frame_start is not None else 0
        end = self.config.frame_end if self.config.frame_end is not None else num_frames - 1
        start = max(0, start)
        end = min(num_frames - 1, end)

        ext = "ply" if self.config.output_format == "ply" else "pt"
        class_name = CATEGORY_NAMES[cat_id]
        # <out>/data/<class_name>/<seq_name>/frame_XXXX.{ply,pt}
        seq_output_dir = os.path.join(self.config.output_dir, "data",
                                      class_name, seq_name)
        os.makedirs(seq_output_dir, exist_ok=True)

        if pbar is not None:
            pbar.set_postfix_str(
                f"{seq_info['category']}/{seq_name}", refresh=False)

        output_files = []

        for frame_idx in range(start, end + 1):
            if self.config.output_format == "ply":
                fname = f"frame_{frame_idx:04d}.{ext}"
            else:
                fname = f"frame_{frame_idx:04d}_s0.{ext}"

            out_path = os.path.join(seq_output_dir, fname)
            rel_path = f"{class_name}/{seq_name}/{fname}"

            if self.config.skip_existing and os.path.isfile(out_path):
                self.stats.frames_skipped += 1
                output_files.append(rel_path)
                if pbar is not None:
                    pbar.update(1)
                continue

            try:
                gaussians = self.process_frame(seq_path, frame_idx)
                if gaussians is None:
                    self.stats.frames_failed += 1
                    tqdm.write(
                        f"[WARN] Missing views: {seq_name} frame {frame_idx}")
                    if pbar is not None:
                        pbar.update(1)
                    continue

                if self.config.output_format == "ply":
                    self.model.gs.save_ply(gaussians, out_path)
                else:
                    save_gaussian_pt(gaussians, out_path)

                self.stats.frames_saved += 1
                output_files.append(rel_path)

            except Exception as e:
                self.stats.frames_failed += 1
                tqdm.write(f"[ERROR] {seq_name} frame {frame_idx}: {e}")

            if pbar is not None:
                pbar.update(1)

        return output_files

    def write_categories(self) -> None:
        """Write data/category.txt (id,name). No split manifests: preprocessing splits via assets/splits/train-test-val_split.yaml."""
        data_dir = os.path.join(self.config.output_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        with open(os.path.join(data_dir, "category.txt"), "w") as f:
            for cat_id in sorted(CATEGORY_NAMES.keys()):
                f.write(f"{cat_id},{CATEGORY_NAMES[cat_id]}\n")

    def profile(self, n_warmup: int = 5, n_runs: int = 50,
                json_out: Optional[str] = None):
        """Per-stage latency on first frame: n_warmup + n_runs, CUDA-synced medians."""
        self.load_model()
        device = torch.device(self.config.device)

        sequences = self.discover_sequences()
        if not sequences:
            sys.exit("[ERROR] No completed sequences found for profiling")

        seq_path = sequences[0]["path"]
        frame_idx = 0
        if self.config.frame_start is not None:
            frame_idx = self.config.frame_start

        views_np = self._load_views(seq_path, frame_idx)
        if views_np is None:
            sys.exit(f"[ERROR] Missing views for frame {frame_idx}")

        print(f"\n[PROFILE] Sequence: {sequences[0]['name']}")
        print(f"[PROFILE] Frame: {frame_idx}")
        print(f"[PROFILE] {n_warmup} warmup + {n_runs} timed runs, "
              f"CUDA-synchronized\n")

        import tempfile
        tmp_ply = os.path.join(tempfile.gettempdir(), "_lgm_profile.ply")

        t_preprocess = []
        t_forward = []
        t_prune = []
        t_serialize = []

        for i in range(n_warmup + n_runs):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()

            input_combined = self._preprocess_views(views_np)

            torch.cuda.synchronize(device)
            t1 = time.perf_counter()

            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    gaussians = self.model.forward_gaussians(input_combined)

            torch.cuda.synchronize(device)
            t2 = time.perf_counter()

            g = gaussians[0]
            opacity = g[:, 3:4]
            valid = opacity.squeeze(-1) >= 0.005
            g_pruned = g[valid]

            torch.cuda.synchronize(device)
            t3 = time.perf_counter()

            self.model.gs.save_ply(gaussians, tmp_ply)

            torch.cuda.synchronize(device)
            t4 = time.perf_counter()

            if i >= n_warmup:
                t_preprocess.append((t1 - t0) * 1000)
                t_forward.append((t2 - t1) * 1000)
                t_prune.append((t3 - t2) * 1000)
                t_serialize.append((t4 - t3) * 1000)

        if os.path.isfile(tmp_ply):
            os.remove(tmp_ply)

        med_pre = float(np.median(t_preprocess))
        med_fwd = float(np.median(t_forward))
        med_prn = float(np.median(t_prune))
        med_ser = float(np.median(t_serialize))
        med_total = med_pre + med_fwd + med_prn + med_ser

        n_gauss_total = 4 * self.opt.splat_size ** 2
        n_gauss_pruned = int(g_pruned.shape[0])

        print("=" * 64)
        print("  LGM Per-Frame Reconstruction Latency (median, ms)")
        print(f"  {n_runs} runs, {n_warmup} warmup, CUDA-synchronized")
        print(f"  Device: {torch.cuda.get_device_name(device)}")
        print("=" * 64)

        rows = [
            ("Preprocess", med_pre),
            ("U-Net forward", med_fwd),
            ("Opacity pruning", med_prn),
            ("Serialization", med_ser),
        ]

        print(f"  {'Sub-stage':<20s} {'Time (ms)':>10s} {'Share (%)':>10s}")
        print(f"  {'-'*20} {'-'*10} {'-'*10}")
        for name, ms in rows:
            pct = ms / med_total * 100
            print(f"  {name:<20s} {ms:>10.2f} {pct:>9.1f}%")
        print(f"  {'-'*20} {'-'*10} {'-'*10}")
        print(f"  {'TOTAL (per frame)':<20s} {med_total:>10.2f} {'100.0%':>10s}")

        fps = 1000.0 / med_total
        print(f"\n  Throughput : {fps:.1f} frames/s")
        print(f"  Gaussians : {n_gauss_total} raw -> {n_gauss_pruned} after pruning")

        n_seq = 73
        n_frames = 80
        total_s = n_seq * n_frames * med_total / 1000
        print(f"\n  Full dataset estimate ({n_seq} seq x {n_frames} frames):")
        print(f"    {total_s:.0f}s ({total_s/60:.1f} min)")

        print(f"\n{'=' * 64}")
        print("  LaTeX table values (copy-paste into main.tex):")
        print("=" * 64)
        for name, ms in rows:
            pct = ms / med_total * 100
            print(f"      & {name:<20s} & {ms:.2f} & {pct:.1f} \\\\")
        print("    \\cmidrule(lr){2-4}")
        print(f"      & {{\\textbf{{Subtotal (per frame)}}}} "
              f"& {med_total:.2f} & -- \\\\")
        print("=" * 64)

        # shape emit_tex reads under 'lgm' in latency_breakdown.json (tab:pipeline-latency)
        if json_out:
            keys = ["preprocess", "unet", "prune", "serialize"]
            meds = [med_pre, med_fwd, med_prn, med_ser]
            samples = [t_preprocess, t_forward, t_prune, t_serialize]
            result = {
                "stages": {
                    k: {"median_ms": m,
                        "mean_ms": float(np.mean(x)),
                        "std_ms": float(np.std(x)),
                        "pct": m / med_total * 100}
                    for k, m, x in zip(keys, meds, samples)
                },
                "total_median_ms": med_total,
                "fps": fps,
                "n_gaussians_raw": n_gauss_total,
                "n_gaussians_pruned": n_gauss_pruned,
                "num_runs": n_runs,
                "num_warmup": n_warmup,
                "device": torch.cuda.get_device_name(device),
                "sequence": sequences[0]["name"],
                "frame": frame_idx,
                "checkpoint": self.config.checkpoint,
            }
            os.makedirs(os.path.dirname(os.path.abspath(json_out)),
                        exist_ok=True)
            with open(json_out, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n[PROFILE] Wrote {json_out}")

    def run(self):
        """Discover, reconstruct all, write category.txt."""
        self.load_model()

        print(f"\n[DISCOVER] Scanning {self.config.input_dir} ...")
        sequences = self.discover_sequences()
        self.stats.total_sequences = len(sequences)

        if not sequences:
            print("[WARN] No completed sequences found "
                  "(missing .render_complete marker)")
            return

        by_cat = {}
        for s in sequences:
            by_cat.setdefault(s["category"], []).append(s["name"])
        for cat, names in sorted(by_cat.items()):
            print(f"  {cat}: {len(names)} sequences")
        print(f"  Total: {len(sequences)} sequences\n")

        # progress bar total
        total_frames = 0
        for seq_info in sequences:
            num = self.get_num_frames(seq_info["path"])
            start = self.config.frame_start if self.config.frame_start is not None else 0
            end = self.config.frame_end if self.config.frame_end is not None else num - 1
            start = max(0, start)
            end = min(num - 1, end)
            total_frames += max(0, end - start + 1)

        pbar = tqdm(total=total_frames, unit="frame",
                    desc="Reconstructing", dynamic_ncols=True)

        for idx, seq_info in enumerate(sequences, 1):
            self.process_sequence(seq_info, pbar=pbar)

        pbar.close()
        print()
        self.write_categories()

        elapsed = self.stats.elapsed
        saved = self.stats.frames_saved
        skipped = self.stats.frames_skipped
        total_output = saved + skipped
        fps = saved / elapsed if elapsed > 0 and saved > 0 else 0
        ms_per_frame = (elapsed / saved * 1000) if saved > 0 else 0

        print(f"\n{'=' * 64}")
        print("  Batch reconstruction complete")
        print(f"  Device    : {torch.cuda.get_device_name()}")
        print(f"  Sequences : {self.stats.total_sequences}")
        print(f"  Saved     : {saved}")
        print(f"  Skipped   : {skipped}")
        print(f"  Failed    : {self.stats.frames_failed}")
        print(f"  Total out : {total_output} files")
        print(f"  Time      : {elapsed:.1f}s")
        if saved > 0:
            print(f"  Throughput: {fps:.1f} frames/s "
                  f"({ms_per_frame:.1f} ms/frame)")
        print(f"  Output    : {self.config.output_dir}")
        print(f"{'=' * 64}")
        if saved > 0:
            print("\n  For supplementary materials:")
            print(f"  - Reconstruction throughput: {fps:.1f} frames/s "
                  f"({ms_per_frame:.1f} ms/frame)")
            print(f"  - Total dataset: {self.stats.total_sequences} sequences, "
                  f"{total_output} 3DGS point clouds")
            print(f"  - Wall-clock time: {elapsed:.1f}s "
                  f"({elapsed/60:.1f} min)")
            n_gauss = 4 * self.opt.splat_size ** 2
            print(f"  - Gaussians per frame: {n_gauss} (before opacity pruning)")
            print("\n  Run with --profile for per-sub-stage latency breakdown.")


def parse_frame_range(s: str) -> tuple[int, int]:
    """'start-end' or single frame."""
    parts = s.split("-")
    if len(parts) == 1:
        n = int(parts[0])
        return n, n
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])
    raise argparse.ArgumentTypeError(f"Invalid frame range: {s!r}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch reconstruct aeroSplat-4D LGM renders into 3DGS")

    parser.add_argument("--input", "-i",
                        default="${AEROSPLAT_RENDER_ROOT}/aeroSplat-4D-LGM",
                        help="Input render directory")
    parser.add_argument("--output", "-o",
                        default="${AEROSPLAT_RENDER_ROOT}/"
                                "aeroSplat-4D-LGM_ply",
                        help="Output directory")
    parser.add_argument("--checkpoint", "-c",
                        default=os.path.join(
                            LGM_ROOT, "pretrained",
                            "model_fp16_fixrot.safetensors"),
                        help="LGM checkpoint path")
    parser.add_argument("--frames", "-f", type=str, default=None,
                        help='Frame range, e.g. "0-5" or "42" (default: all)')
    parser.add_argument("--format", choices=["ply", "torch"], default="ply",
                        help="Output format: ply (default) or torch (.pt)")
    parser.add_argument("--no-skip", action="store_true",
                        help="Reprocess existing outputs")
    parser.add_argument("--device", default="cuda",
                        help="Device (default: cuda)")
    parser.add_argument("--border-ratio", type=float, default=0.2,
                        help="Crop border padding (default: 0.2)")
    parser.add_argument("--profile", action="store_true",
                        help="Profile per-sub-stage latency "
                             "(5 warmup + 50 timed runs)")
    parser.add_argument("--profile-warmup", type=int, default=5,
                        help="Number of warmup iterations (default: 5)")
    parser.add_argument("--profile-runs", type=int, default=50,
                        help="Number of timed iterations (default: 50)")
    parser.add_argument("--profile-json", type=str, default=None,
                        help="Also write the --profile result as JSON "
                             "(the 'lgm' block of latency_breakdown.json)")

    args = parser.parse_args()

    config = BatchConfig(
        input_dir=args.input,
        output_dir=args.output,
        checkpoint=args.checkpoint,
        output_format=args.format,
        skip_existing=not args.no_skip,
        device=args.device,
        border_ratio=args.border_ratio,
    )

    if args.frames:
        config.frame_start, config.frame_end = parse_frame_range(args.frames)

    reconstructor = BatchReconstructor(config)

    if args.profile:
        reconstructor.profile(n_warmup=args.profile_warmup,
                              n_runs=args.profile_runs,
                              json_out=args.profile_json)
    else:
        reconstructor.run()


if __name__ == "__main__":
    main()
