#!/usr/bin/env python3

"""Batch render orchestrator (host Python).

Reads configs/main.yaml + configs/config_batch.yaml, expands sweep to one job per
(asset, camera rig, animation speed), writes JSON manifest, launches
runners/smart_batch_runner.py in Isaac Sim for jobs lacking `.render_complete`.

    python batch_render.py --dry-run                 # list jobs
    python batch_render.py                           # render missing
    python batch_render.py --asset-type bird         # one class
    python batch_render.py --assets "crow.usdc,DJI Inspire 3.usdc"
    python batch_render.py --no-resume               # re-render finished
    python batch_render.py --isaac-sim-path ~/isaacsim
"""

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).parent.resolve()
STAGE_ROOT = SCRIPT_DIR.parent
CONFIGS_DIR = STAGE_ROOT / "configs"

if str(STAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE_ROOT))

from utils.config_utils import deep_merge, expand_env, load_main_config, load_merged_config  # noqa: E402

COMPLETE_MARKER = ".render_complete"


def load_batch_config(config_path=None):
    """Batch YAML deep-merged with main.yaml."""
    if config_path is None:
        cfg = load_merged_config(CONFIGS_DIR, "config_batch.yaml")
        if cfg is None:
            print(f"[BATCH] ERROR: Batch config not found: {CONFIGS_DIR / 'config_batch.yaml'}")
        return cfg

    config_path = Path(config_path)
    if not config_path.exists():
        print(f"[BATCH] ERROR: Batch config not found: {config_path}")
        return None

    with open(config_path, "r") as f:
        mode_cfg = expand_env(yaml.safe_load(f))

    base = load_main_config(CONFIGS_DIR)
    return deep_merge(base, mode_cfg) if isinstance(mode_cfg, dict) else base


def load_asset_config():
    """asset_config.yaml: USD paths and types."""
    asset_config_path = CONFIGS_DIR / "asset_config.yaml"
    if not asset_config_path.exists():
        print(f"[BATCH] ERROR: Asset config not found: {asset_config_path}")
        return None

    with open(asset_config_path, "r") as f:
        return expand_env(yaml.safe_load(f))


def get_available_assets(asset_config):
    """[{name, type, usd_path}] for all assets."""
    assets = []
    for name, data in asset_config.items():
        assets.append({
            "name": name,
            "type": data.get("type", "unknown"),
            "usd_path": data.get("usd_path", ""),
        })
    return assets


def filter_assets(assets, batch_cfg, args):
    """Filter by batch config, --assets, --asset-type."""
    config_assets = batch_cfg.get("assets", [])

    if config_assets == "all" or (len(config_assets) == 1 and config_assets[0] == "all"):
        filtered = [a["name"] for a in assets]
    else:
        available_names = {a["name"] for a in assets}
        filtered = []
        for asset_name in config_assets:
            if asset_name in available_names:
                filtered.append(asset_name)
            else:
                print(f"[BATCH] WARNING: Asset '{asset_name}' not found in asset_config.yaml")

    if args.assets:
        requested = [a.strip() for a in args.assets.split(",")]
        filtered = [a for a in filtered if a in requested]

    if args.asset_type:
        asset_types = {a["name"]: a["type"] for a in assets}
        filtered = [a for a in filtered if asset_types.get(a) == args.asset_type]

    return filtered


def _resolve_flight_direction(direction_cfg, seed_key, elevation_range=None):
    """[dx, dy, dz] from fixed list or "random" (sha256(seed_key), reproducible). elevation_range [min_deg, max_deg], default horizontal."""
    if isinstance(direction_cfg, str) and direction_cfg.lower() == "random":
        digest = hashlib.sha256(seed_key.encode("utf-8")).digest()
        int_val = int.from_bytes(digest[:8], "big")
        az = (int_val % (2 ** 32)) / (2 ** 32) * 2 * math.pi
        el = 0.0
        if elevation_range and (elevation_range[0] != 0 or elevation_range[1] != 0):
            # next 8 bytes: azimuth stays stable
            int_val2 = int.from_bytes(digest[8:16], "big")
            el_deg = elevation_range[0] + (int_val2 % (2 ** 32)) / (2 ** 32) * (elevation_range[1] - elevation_range[0])
            el = math.radians(el_deg)
        return [round(math.cos(el) * math.cos(az), 6),
                round(math.cos(el) * math.sin(az), 6),
                round(math.sin(el), 6)]
    return list(direction_cfg)


def generate_combinations(batch_cfg, assets, all_assets):
    """All sweep combinations. `lgm.enabled: true` = four-view LGM rig, ignores `cameras.*`; else `cameras.placement` picks sphere/circle."""
    cam_cfg = batch_cfg.get("cameras", {})
    circle_cfg = cam_cfg.get("circle", {})
    sphere_cfg = cam_cfg.get("sphere", {})

    camera_placement = cam_cfg.get("placement", "circle")
    camera_types = batch_cfg.get("camera_types", ["ip_cam_2k"])
    animation_speed_factors = batch_cfg.get("animation_speed_factors", [1.0])

    # single values, not sweep lists
    waypoint = batch_cfg.get("waypoint", [0.0, 0.0, 0.0])
    flight_distance = batch_cfg.get("flight_distance", 10.0)
    flight_direction = batch_cfg.get("flight_direction", [1.0, 0.0, 0.0])
    random_elevation_range = batch_cfg.get("random_elevation_range", [0, 0])

    asset_types = {a["name"]: a["type"] for a in all_assets}

    combinations = []

    lgm_cfg = batch_cfg.get("lgm", {})
    if lgm_cfg.get("enabled", False):
        lgm_radii = lgm_cfg.get("radii", [25.0])
        lgm_elevations = lgm_cfg.get("elevations", [0])
        lgm_center = lgm_cfg.get("center", [0.0, 0.0, 0.0])
        _raw_az = lgm_cfg.get("azimuth_offset", 0.0)
        lgm_azimuth_offsets = [float(v) for v in (_raw_az if isinstance(_raw_az, list) else [_raw_az])]
        lgm_resolution = list(lgm_cfg.get("resolution", [256, 256]))

        # static cameras; diversity from random flight direction + nose rotation
        for asset_name, lgm_r, lgm_el, lgm_az, anim_speed in itertools.product(
            assets, lgm_radii, lgm_elevations, lgm_azimuth_offsets, animation_speed_factors
        ):
            combinations.append({
                "asset": asset_name,
                "asset_type": asset_types.get(asset_name, "unknown"),
                "camera_placement": "lgm",
                "lgm_radius": lgm_r,
                "lgm_elevation": lgm_el,
                "lgm_center": lgm_center,
                "lgm_azimuth_offset": lgm_az,
                "lgm_resolution": lgm_resolution,
                "camera_type": "lgm_pinhole",
                "animation_speed_factor": anim_speed,
                "waypoint": waypoint,
                "flight_distance": flight_distance,
                "flight_direction": _resolve_flight_direction(
                    flight_direction,
                    f"{asset_name}|blank|lgm_{lgm_r}_{lgm_el}_{lgm_az}_{anim_speed}",
                    elevation_range=random_elevation_range),
            })
        return combinations

    if camera_placement == "sphere":
        sphere_radii = sphere_cfg.get("radii", [10.0])
        sphere_num_cameras = int(sphere_cfg.get("num_cameras", 20))
        sphere_center = sphere_cfg.get("center", [0.0, 0.0, 0.0])
        sphere_hemisphere = sphere_cfg.get("hemisphere", "full")

        for asset_name, sph_r, cam_type, anim_speed in itertools.product(
            assets, sphere_radii, camera_types, animation_speed_factors
        ):
            combinations.append({
                "asset": asset_name,
                "asset_type": asset_types.get(asset_name, "unknown"),
                "camera_placement": "sphere",
                "sphere_radius": sph_r,
                "sphere_num_cameras": sphere_num_cameras,
                "sphere_center": sphere_center,
                "sphere_hemisphere": sphere_hemisphere,
                "camera_type": cam_type,
                "animation_speed_factor": anim_speed,
                "waypoint": waypoint,
                "flight_distance": flight_distance,
                "flight_direction": _resolve_flight_direction(
                    flight_direction,
                    f"{asset_name}|blank|sphere_{sph_r}_{cam_type}_{anim_speed}",
                    elevation_range=random_elevation_range),
            })
        return combinations

    side_meters = circle_cfg.get("side_meters", [10.0])
    cam_heights = circle_cfg.get("cam_heights", [7.0])
    rotation_offsets = circle_cfg.get("rotation_offsets", [0.0])
    aim_point_height_offsets = circle_cfg.get("aim_point_height_offsets", [-5.0])
    circle_num_cameras = int(circle_cfg.get("num_cameras", 5))

    for asset_name, side, cam_h, rot, aim_h, cam_type, anim_speed in itertools.product(
        assets, side_meters, cam_heights, rotation_offsets, aim_point_height_offsets,
        camera_types, animation_speed_factors
    ):
        combinations.append({
            "asset": asset_name,
            "asset_type": asset_types.get(asset_name, "unknown"),
            "camera_placement": "circle",
            "num_cameras": circle_num_cameras,
            "side_meters": side,
            "cam_height": cam_h,
            "rotation_offset": rot,
            "aim_point_height_offset": aim_h,
            "camera_type": cam_type,
            "animation_speed_factor": anim_speed,
            "waypoint": waypoint,
            "flight_distance": flight_distance,
            "flight_direction": _resolve_flight_direction(
                flight_direction,
                f"{asset_name}|blank|circle_{side}_{cam_h}_{rot}_{aim_h}_{cam_type}_{anim_speed}",
                elevation_range=random_elevation_range),
        })

    return combinations


def get_output_dir(combo, batch_cfg):
    """Output dir for combo. LGM folder name is parsed downstream: format is dataset interface."""
    base_path = batch_cfg.get("output", {}).get("base_path", "renders")

    asset_clean = Path(combo["asset"]).stem
    asset_clean = re.sub(r'[^\w\-]', '_', asset_clean)

    if combo.get("camera_placement") == "lgm":
        folder_name = "blank_lgm_{radius}r_{el}el_{az}az_4cams_lgm_pinhole_{anim_speed}x_{asset}".format(
            asset=asset_clean,
            radius=combo.get("lgm_radius", 25),
            el=int(combo.get("lgm_elevation", 0)),
            az=int(combo.get("lgm_azimuth_offset", 0)),
            anim_speed=combo.get("animation_speed_factor", 1.0),
        )
    elif combo.get("camera_placement") == "sphere":
        folder_name = "blank_sphere_{radius}r_{ncams}cams_{cam_type}_{anim_speed}x_{asset}".format(
            asset=asset_clean,
            radius=int(combo["sphere_radius"]),
            ncams=int(combo.get("sphere_num_cameras", 20)),
            cam_type=combo["camera_type"],
            anim_speed=combo.get("animation_speed_factor", 1.0),
        )
    else:
        folder_name = "blank_circle_{side}m_{ncams}cams_{cam_type}_{anim_speed}x_{asset}".format(
            asset=asset_clean,
            side=int(combo["side_meters"]),
            ncams=int(combo.get("num_cameras", 5)),
            cam_type=combo["camera_type"],
            anim_speed=combo.get("animation_speed_factor", 1.0),
        )

    return os.path.join(base_path, combo.get("asset_type", "unknown"), folder_name)


def is_already_rendered(output_dir):
    """Completion marker present."""
    return (Path(output_dir) / COMPLETE_MARKER).exists()


def find_isaac_sim_python(isaac_sim_path=None):
    """python.sh from --isaac-sim-path, ISAAC_SIM_PATH, or ~/isaacsim."""
    for path in (isaac_sim_path, os.environ.get("ISAAC_SIM_PATH"), os.path.expanduser("~/isaacsim")):
        if not path:
            continue
        python_sh = Path(path) / "python.sh"
        if python_sh.exists():
            return str(python_sh)
    return None


def _cleanup_after_crash(wait_seconds=10):
    """Wait for GPU release after crash; kill lingering Kit processes."""
    print(f"[BATCH] Waiting {wait_seconds}s for GPU resources to release...")
    time.sleep(wait_seconds)
    for pattern in ["omni.kit", "isaac-sim"]:
        try:
            subprocess.run(
                ["pkill", "-f", pattern],
                check=False, capture_output=True, timeout=5,
            )
        except Exception:
            pass
    time.sleep(2)


def build_manifest(combinations, batch_cfg, permanently_failed=()):
    """Job list for smart runner; finished/abandoned flagged already_rendered."""
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "total_combinations": len(combinations),
        "render_settings": batch_cfg.get("render", {}),
        "combinations": [],
    }
    for combo in combinations:
        output_dir = get_output_dir(combo, batch_cfg)
        manifest["combinations"].append({
            **combo,
            "output_dir": output_dir,
            "already_rendered": is_already_rendered(output_dir) or output_dir in permanently_failed,
        })
    return manifest


def write_manifest(manifest, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)


def describe_job(combo):
    placement = combo.get("camera_placement")
    if placement == "lgm":
        rig = f"lgm r={combo['lgm_radius']}m el={combo['lgm_elevation']} az={combo['lgm_azimuth_offset']}"
    elif placement == "sphere":
        rig = f"sphere r={combo['sphere_radius']}m cams={combo['sphere_num_cameras']} {combo['camera_type']}"
    else:
        rig = f"circle side={combo['side_meters']}m h={combo['cam_height']}m cams={combo['num_cameras']} {combo['camera_type']}"
    return f"{combo['asset_type']:10} {rig}  speed={combo['animation_speed_factor']}x"


def main():
    parser = argparse.ArgumentParser(
        description="Batch rendering orchestrator for Isaac Sim",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config_batch.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Print the jobs without rendering")
    parser.add_argument("--assets", type=str, default=None, help="Comma-separated list of specific assets to render")
    parser.add_argument("--asset-type", type=str, choices=["bird", "drone", "helicopter", "airplane"],
                        help="Filter assets by type")
    parser.add_argument("--isaac-sim-path", type=str, default=None, help="Path to Isaac Sim installation")
    parser.add_argument("--no-resume", action="store_true",
                        help="Remove the completion markers of the selected jobs so they render again")
    parser.add_argument("--list-assets", action="store_true", help="List available assets and exit")
    parser.add_argument("--save-manifest", type=str, default=None, help="Write the job manifest to this JSON file")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("Isaac Sim Batch Rendering Orchestrator")
    print("=" * 70)

    batch_cfg = load_batch_config(args.config)
    if batch_cfg is None:
        sys.exit(1)

    asset_config = load_asset_config()
    if asset_config is None:
        sys.exit(1)

    all_assets = get_available_assets(asset_config)

    if args.list_assets:
        print("\nAvailable assets:")
        for asset in sorted(all_assets, key=lambda x: (x["type"], x["name"])):
            print(f"  [{asset['type']:10}] {asset['name']}")
        sys.exit(0)

    assets_to_render = filter_assets(all_assets, batch_cfg, args)
    if not assets_to_render:
        print("[BATCH] ERROR: No assets to render after filtering")
        sys.exit(1)

    print(f"\n[BATCH] Assets to render: {len(assets_to_render)}")
    for asset in assets_to_render:
        print(f"  - {asset}")

    combinations = generate_combinations(batch_cfg, assets_to_render, all_assets)
    print(f"\n[BATCH] Total combinations: {len(combinations)}")

    if args.no_resume:
        removed = 0
        for combo in combinations:
            marker = Path(get_output_dir(combo, batch_cfg)) / COMPLETE_MARKER
            if marker.exists():
                marker.unlink()
                removed += 1
        print(f"[BATCH] --no-resume: removed {removed} completion markers")

    if args.dry_run:
        print("\n[BATCH] DRY RUN - jobs that would be rendered:")
        done = 0
        for combo in combinations:
            output_dir = get_output_dir(combo, batch_cfg)
            finished = is_already_rendered(output_dir)
            done += finished
            print(f"  [{'done' if finished else 'todo'}] {describe_job(combo)}\n         -> {output_dir}")
        print(f"\n[BATCH] {len(combinations) - done} to render, {done} already finished")
        if args.save_manifest:
            write_manifest(build_manifest(combinations, batch_cfg), args.save_manifest)
            print(f"[BATCH] Manifest saved to: {args.save_manifest}")
        sys.exit(0)

    isaac_python = find_isaac_sim_python(args.isaac_sim_path)
    if isaac_python is None:
        print("[BATCH] ERROR: Could not find Isaac Sim Python interpreter")
        print("[BATCH] Set ISAAC_SIM_PATH environment variable or use --isaac-sim-path")
        sys.exit(1)
    print(f"[BATCH] Using Isaac Sim Python: {isaac_python}")

    if args.save_manifest:
        manifest_path = Path(args.save_manifest)
    else:
        base_path = batch_cfg.get("output", {}).get("base_path", "renders")
        manifest_path = Path(base_path) / f"batch_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    base_cmd = [isaac_python, str(SCRIPT_DIR / "smart_batch_runner.py"), "--manifest", str(manifest_path)]
    runner_env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    print("\n" + "-" * 70)
    print("Starting batch rendering...")
    print("[BATCH] One Isaac Sim process renders all jobs; on a crash it is")
    print("[BATCH] restarted, the crashed job retried once, then skipped.")
    print("-" * 70)

    retried_dirs = set()
    permanently_failed_dirs = set()
    max_restarts = len(combinations) + 5
    restart_count = 0

    try:
        while restart_count <= max_restarts:
            manifest = build_manifest(combinations, batch_cfg, permanently_failed_dirs)
            pending = [c for c in manifest["combinations"] if not c["already_rendered"]]
            if not pending:
                print("[BATCH] All jobs completed (or permanently failed).")
                break

            write_manifest(manifest, manifest_path)
            print(f"\n[BATCH] Launching smart runner ({len(pending)} jobs remaining)...")
            result = subprocess.run(base_cmd, check=False, env=runner_env)

            if result.returncode == 0:
                print("[BATCH] Smart runner finished.")
                break

            restart_count += 1
            print(f"\n[BATCH] {'=' * 60}")
            print(f"[BATCH] CRASH DETECTED (exit code {result.returncode})")
            print(f"[BATCH] {'=' * 60}")
            _cleanup_after_crash()

            # runner is in-order: first unmarked pending job crashed
            crashed = next((c for c in pending if not is_already_rendered(c["output_dir"])), None)
            if crashed is None:
                print("[BATCH] All jobs appear completed despite crash exit code.")
                break

            out = crashed["output_dir"]
            if out in retried_dirs:
                permanently_failed_dirs.add(out)
                print(f"[BATCH] PERMANENTLY FAILED (crashed twice): {crashed['asset']}")
                print(f"[BATCH]   Output: {out}")
            else:
                retried_dirs.add(out)
                print(f"[BATCH] Will RETRY on next launch: {crashed['asset']}")
                if Path(out).exists():
                    shutil.rmtree(out, ignore_errors=True)
                    print(f"[BATCH] Cleaned partial output: {out}")
        else:
            print(f"[BATCH] Max restarts ({max_restarts}) exceeded. Stopping.")

    except KeyboardInterrupt:
        print("\n[BATCH] Interrupted by user")
        sys.exit(130)

    unfinished = [get_output_dir(c, batch_cfg) for c in combinations
                  if not is_already_rendered(get_output_dir(c, batch_cfg))]

    print(f"\n{'=' * 70}")
    print("Batch Rendering Complete")
    print(f"{'=' * 70}")
    print(f"  Total combinations:  {len(combinations)}")
    print(f"  Successful:          {len(combinations) - len(unfinished)}")
    print(f"  Not rendered:        {len(unfinished)}")
    for d in unfinished:
        print(f"    - {d}" + ("  (crashed twice)" if d in permanently_failed_dirs else ""))
    print(f"  Isaac Sim restarts:  {restart_count}")
    print(f"{'=' * 70}\n")

    sys.exit(0 if not unfinished else 1)


if __name__ == "__main__":
    main()
