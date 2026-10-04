"""End-to-end pipeline latency breakdown (supplementary Tab. 9), printed as numbers.

    python -m mambasplat4d.cli.latency experiment=aerosplat4d
    python -m mambasplat4d.cli.latency experiment=aerosplat4d dataset.feature_mode=C \\
        +lgm_json=/path/to/lgm_latency.json    # from pipelines/lgm/batch_reconstruct.py --profile-json

Random weights and inputs at the training operating point (B=1, dataset.num_points,
dataset.seq_len), fp32, no torch.compile. Plain timed passes and hooked passes alternate;
the hooked pass records CUDA events at the stage boundaries of one unmodified forward,
so the four stages add up to the end-to-end time.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from mambasplat4d import paths
from mambasplat4d.models.pipeline import ClassificationPipeline
from mambasplat4d.utils.feature_config import FEATURE_MODES

STAGES = ("gaussian_lifting", "vn_transformer", "vn_in", "mamba")
BOUNDARIES = ("start", "encoder_in", "encoder_out", "bridge_out", "end")
LGM_STAGES = ("preprocess", "unet", "prune", "serialize")
KEY_DIMS = {"position": 3, "opacity": 1, "scale": 3, "quaternion": 4, "sh_dc": 3}


def measure(cfg: DictConfig, runs: int, warmup: int) -> dict:
    torch.manual_seed(0)
    model = ClassificationPipeline(cfg).cuda().eval()
    N, T = int(cfg.dataset.num_points), int(cfg.dataset.seq_len)
    keys = FEATURE_MODES[cfg.dataset.get("feature_mode", "X")]
    inp = {
        "gaussian_data": {
            k: torch.randn(T, N, KEY_DIMS[k], device="cuda") for k in keys
        },
        "mask": torch.ones(T, N, dtype=torch.bool, device="cuda"),
        "T": T,
    }

    events: dict = {}
    hooks_on = [False]

    def mark(name):
        def hook(*_args, **_kwargs):
            if hooks_on[0]:
                events[name] = torch.cuda.Event(enable_timing=True)
                events[name].record()

        return hook

    encoder = model.spatial_encoder.encoder
    bridge = model.temporal_encoder.bridge
    handles = [
        model.register_forward_pre_hook(mark("start")),
        encoder.register_forward_pre_hook(mark("encoder_in")),
        encoder.register_forward_hook(mark("encoder_out")),
        bridge.register_forward_hook(mark("bridge_out")),
        model.register_forward_hook(mark("end")),
    ]

    stage_ms = {s: [] for s in STAGES}
    plain_ms = []
    with torch.no_grad():
        for _ in range(warmup):
            model(**inp)
        torch.cuda.synchronize()
        for _ in range(runs):
            hooks_on[0] = False
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(**inp)
            torch.cuda.synchronize()
            plain_ms.append((time.perf_counter() - t0) * 1000)

            hooks_on[0] = True
            events.clear()
            model(**inp)
            torch.cuda.synchronize()
            for s, a, b in zip(STAGES, BOUNDARIES[:-1], BOUNDARIES[1:]):
                stage_ms[s].append(events[a].elapsed_time(events[b]))
    for h in handles:
        h.remove()

    stages = {s: statistics.median(v) for s, v in stage_ms.items()}
    return {
        "stages": stages,
        "total_median_ms": sum(stages.values()),
        "end_to_end_median_ms": statistics.median(plain_ms),
        "N": N,
        "T": T,
        "feature_mode": cfg.dataset.get("feature_mode", "X"),
        "gpu": torch.cuda.get_device_name(0),
        "runs": runs,
        "warmup": warmup,
    }


def report(r: dict, lgm: dict | None) -> None:
    print(
        f"GPU {r['gpu']}  N={r['N']}  T={r['T']}  E({r['feature_mode']})  "
        f"median of {r['runs']} runs after {r['warmup']} warm-up passes"
    )
    if lgm is not None:
        lt = lgm["total_median_ms"]
        print("LGM reconstruction (per frame)")
        for s in LGM_STAGES:
            v = lgm["stages"][s]["median_ms"]
            print(f"  {s:<18} {v:8.2f} ms  {100 * v / lt:5.1f} %")
        print(f"  {'subtotal':<18} {lt:8.2f} ms")
    total = r["total_median_ms"]
    print(f"MambaSplat-4D (T={r['T']}, single pass)")
    for s in STAGES:
        v = r["stages"][s]
        print(f"  {s:<18} {v:8.2f} ms  {100 * v / total:5.1f} %")
    print(f"  {'subtotal':<18} {total:8.2f} ms")
    print(f"  {'end-to-end':<18} {r['end_to_end_median_ms']:8.2f} ms")
    if lgm is not None:
        clip = r["T"] * lgm["total_median_ms"] + total
        print(
            f"Clip total (T x LGM + classify) {clip:8.1f} ms  "
            f"classifier share {100 * total / clip:.1f} %"
        )


@hydra.main(config_path=paths.CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("latency breakdown needs a CUDA GPU")
    lat = cfg.get("latency", {}) or {}
    r = measure(cfg, runs=int(lat.get("runs", 200)), warmup=int(lat.get("warmup", 20)))
    lgm_path = cfg.get("lgm_json", None)
    lgm = json.loads(Path(str(lgm_path)).read_text()) if lgm_path else None
    report(r, lgm)
    out = cfg.get("out", None)
    if out:
        if lgm is not None:
            r["lgm"] = lgm
        Path(str(out)).write_text(json.dumps(r, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
