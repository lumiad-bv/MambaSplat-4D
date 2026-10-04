"""Continuous batch runner inside Isaac Sim (python.sh).

Runs manifest jobs from batch_render.py sequentially in one simulator process:
blank stage once, then per job reload asset, cameras, animation.
"""

# ruff: noqa: E402  (SimulationApp before any omni/isaacsim import)

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
STAGE_ROOT = SCRIPT_DIR.parent


def _resolve_max_resolution_from_manifest():
    """Max camera resolution from --manifest, pre-SimulationApp. Viewport must be >= render product resolution, else corrupt framebuffers (Pillow segfault in ImagingZipEncode)."""
    import yaml as _yaml

    manifest_path = None
    for i, arg in enumerate(sys.argv):
        if arg == "--manifest" and i + 1 < len(sys.argv):
            manifest_path = sys.argv[i + 1]
            break
    if manifest_path is None or not os.path.exists(manifest_path):
        return (1920, 1080)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    cam_types = set()
    for combo in manifest.get("combinations", []):
        ct = combo.get("camera_type")
        if ct:
            cam_types.add(ct)

    if not cam_types:
        return (1920, 1080)

    intrinsics_path = STAGE_ROOT / "configs" / "cam_intrinsics.yaml"
    if not intrinsics_path.exists():
        return (1920, 1080)

    with open(intrinsics_path, "r") as f:
        intrinsics_db = _yaml.safe_load(f)

    max_w, max_h = 1920, 1080
    for ct in cam_types:
        entry = intrinsics_db.get(ct, {})
        orig = entry.get("original_resolution")
        if orig and len(orig) == 2:
            max_w = max(max_w, int(orig[0]))
            max_h = max(max_h, int(orig[1]))

    return (max_w, max_h)


_viewport_res = _resolve_max_resolution_from_manifest()

# SimulationApp first; viewport >= render product resolution
from isaacsim import SimulationApp

CONFIG = {"headless": True, "width": _viewport_res[0], "height": _viewport_res[1]}
simulation_app = SimulationApp(CONFIG)
print(f"[SMART] SimulationApp viewport: {_viewport_res[0]}x{_viewport_res[1]}")

# SimulationApp prepends isaacsim paths that shadow ours (e.g. utils); re-insert STAGE_ROOT first
_stage_str = str(STAGE_ROOT)
if _stage_str in sys.path:
    sys.path.remove(_stage_str)
sys.path.insert(0, _stage_str)

# Kit meta-path finders expose e.g. cv2/utils as top-level "utils"; force-register ours
import importlib.util as _ilu


def _force_register_local_package(name):
    pkg_dir = STAGE_ROOT / name
    init_file = pkg_dir / "__init__.py"
    if not init_file.exists():
        return
    for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
        del sys.modules[key]
    spec = _ilu.spec_from_file_location(
        name,
        str(init_file),
        submodule_search_locations=[str(pkg_dir)],
    )
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)


_force_register_local_package("utils")
_force_register_local_package("core")

import omni.usd
import omni.kit.app
import carb.settings

import core.load_scene as load_scene
import core.load_drone as load_drone
import core.position_cameras as position_cameras
import core.animate as animate
import core.render as render


def load_base_config():
    """main.yaml base config."""
    from utils.config_utils import load_main_config

    return load_main_config(STAGE_ROOT / "configs")


def apply_combo_to_config(base_cfg, combo):
    """Base config + combo params."""
    import copy
    import yaml
    from utils.config_utils import expand_env

    cfg = copy.deepcopy(base_cfg)
    drone = cfg.setdefault("drone", {})

    asset_config_path = STAGE_ROOT / "configs" / "asset_config.yaml"
    asset_paths = {}
    if asset_config_path.exists():
        with open(asset_config_path, "r") as f:
            asset_paths = expand_env(yaml.safe_load(f)) or {}

    asset_name = combo["asset"]
    asset_data = asset_paths.get(asset_name, {})

    # asset config = defaults, combo wins; identity keys always applied
    _ASSET_IDENTITY_KEYS = {"usd_path", "type", "translation", "orientation", "scale"}
    for key, value in asset_data.items():
        if key in _ASSET_IDENTITY_KEYS or key not in drone:
            drone[key] = value
    drone["asset_name"] = asset_name
    drone["prim_path"] = "/World/Drone"

    cams = cfg.setdefault("cameras", {})
    cams["camera_type"] = combo["camera_type"]

    placement = combo.get("camera_placement", "circle")
    if placement == "lgm":
        cams["placement"] = "lgm"
        cams["camera_type"] = "lgm_pinhole"
        lgm_res = combo.get("lgm_resolution", [256, 256])
        cams["resolution"] = list(lgm_res)
        lgm = cfg.setdefault("lgm", {})
        lgm["enabled"] = True
        lgm["radius"] = float(combo.get("lgm_radius", 25.0))
        lgm["elevation"] = float(combo.get("lgm_elevation", 0))
        lgm["azimuth_offset"] = float(combo.get("lgm_azimuth_offset", 0.0))
        lgm["center"] = combo.get("lgm_center", [0.0, 0.0, 0.0])
        lgm["resolution"] = list(lgm_res)
    elif placement == "sphere":
        cams["placement"] = "sphere"
        sphere = cams.setdefault("sphere", {})
        sphere["radius"] = float(combo.get("sphere_radius", 10.0))
        sphere["num_cameras"] = int(combo.get("sphere_num_cameras", 20))
        sphere["center"] = combo.get("sphere_center", [0.0, 0.0, 0.0])
        sphere["hemisphere"] = combo.get("sphere_hemisphere", "full")
    else:
        cams["placement"] = "circle"
        circle = cams.setdefault("circle", {})
        circle["side_meters"] = float(combo["side_meters"])
        circle["cam_height"] = float(combo["cam_height"])
        circle["rotation_offset"] = float(combo["rotation_offset"])
        circle["num_cameras"] = int(combo.get("num_cameras", 5))
        if "aim_point_height_offset" in combo:
            circle["aim_point_height_offset"] = float(combo["aim_point_height_offset"])

    drone["animation_speed_factor"] = float(combo.get("animation_speed_factor", 1.0))
    drone["asset_type"] = combo.get("asset_type", "unknown")
    if "waypoint" in combo:
        drone["waypoint"] = list(combo["waypoint"])
    if "flight_distance" in combo:
        drone["flight_distance"] = float(combo["flight_distance"])
    if "flight_direction" in combo:
        drone["flight_direction"] = list(combo["flight_direction"])

    render_cfg = cfg.setdefault("render", {})
    if cams["placement"] == "lgm":
        render_cfg["num_cameras"] = 4
    else:
        render_cfg["num_cameras"] = int(
            cams.get(cams["placement"], {}).get("num_cameras", 5)
        )
    render_cfg["num_frames"] = int(combo.get("num_frames", render_cfg.get("num_frames", 120)))
    render_cfg["output_dir"] = combo["output_dir"]
    if "passes" in combo:
        render_cfg.setdefault("passes", {}).update(combo["passes"])

    return cfg


def delete_prim(stage, path):
    """Remove prim subtree."""
    if not path:
        return
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        stage.RemovePrim(path)


def cleanup_cameras(stage):
    """Remove /World/cam_* prims from PositionCameras."""
    # collect first: avoids "Iterator points to expired prim"
    paths_to_delete = []
    try:
        for prim in stage.Traverse():
            if prim.GetName().startswith(
                "cam_"
            ) and prim.GetPath().pathString.startswith("/World/cam_"):
                paths_to_delete.append(prim.GetPath())

        for path in paths_to_delete:
            if stage.GetPrimAtPath(path).IsValid():
                stage.RemovePrim(path)

    except Exception as e:
        print(f"[SMART] Warning: Error during camera cleanup: {e}")


def cleanup_replicator(stage):
    """Remove stale Replicator OmniGraph prims.

    Render calls leave prims under /Render and /WriterOrchestrator despite
    rp.destroy()/writer.detach(); they accumulate across jobs and destabilise
    the physics plugin. Never delete /Render/OmniverseKit (Kit viewport).
    """
    import omni.replicator.core as rep

    try:
        rep.orchestrator.stop()
    except Exception:
        pass

    # fully Replicator-owned
    wo_prim = stage.GetPrimAtPath("/WriterOrchestrator")
    if wo_prim.IsValid():
        try:
            stage.RemovePrim("/WriterOrchestrator")
            print("[SMART] Cleaned up /WriterOrchestrator")
        except Exception as exc:
            print(f"[SMART] Warning: Failed to remove /WriterOrchestrator: {exc}")

    # keep Kit-owned OmniverseKit
    render_prim = stage.GetPrimAtPath("/Render")
    if render_prim.IsValid():
        children_to_remove = []
        for child in render_prim.GetChildren():
            if child.GetName() == "OmniverseKit":
                continue
            children_to_remove.append(child.GetPath())

        for path in children_to_remove:
            try:
                stage.RemovePrim(path)
            except Exception as exc:
                print(f"[SMART] Warning: Failed to remove {path}: {exc}")

        if children_to_remove:
            print(
                f"[SMART] Cleaned up {len(children_to_remove)} replicator prims under /Render"
            )


def disable_physics(settings):
    """Disable PhysX updates (render-only). Plugin tracks all xformOps; over 100+ jobs state accumulates and segfaults libomni.usdphysics.plugin.so."""
    physics_keys = [
        ("/physics/autoUpdate", False),
        ("/persistent/physics/updateToUsd", False),
        ("/persistent/physics/updateVelocitiesToUsd", False),
    ]
    for key, value in physics_keys:
        try:
            settings.set(key, value)
        except Exception:
            pass
    print("[SMART] Physics scene updates disabled (render-only mode)")


async def run_smart_batch(manifest_path):
    print(f"[SMART] Loading manifest: {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    combos = manifest.get("combinations", [])
    base_render_cfg = manifest.get("render_settings", {})

    print(f"[SMART] Loaded {len(combos)} jobs. Executing continuously...")

    kit_app = omni.kit.app.get_app()
    base_config = load_base_config()

    if not base_config.get("physics", {}).get("enabled", True):
        disable_physics(carb.settings.get_settings())

    if not load_scene.LoadScene(base_config).run():
        raise RuntimeError("Failed to create blank stage")
    for _ in range(10):
        await kit_app.next_update_async()

    succeeded = 0
    failed = 0
    skipped = 0

    for i, combo in enumerate(combos):
        job_id = f"Job {i + 1}/{len(combos)}"
        print(f"\n{'-' * 60}")
        print(f"[SMART] {job_id}: {combo['asset']}")
        print(f"{'-' * 60}")

        # manifest render settings fill combo gaps
        for key in ("num_frames", "num_cameras", "passes"):
            if key not in combo and key in base_render_cfg:
                combo[key] = base_render_cfg[key]

        if combo.get("already_rendered", False):
            print("[SMART] Skipping (marked as done in manifest)")
            skipped += 1
            continue

        output_dir = combo["output_dir"]
        cfg = apply_combo_to_config(base_config, combo)

        try:
            stage = omni.usd.get_context().get_stage()

            # clean stage: previous asset, mover, cameras, replicator graph
            delete_prim(stage, cfg["drone"]["prim_path"])
            delete_prim(stage, cfg["drone"].get("mover_prim_path", "/World/DroneMover"))
            cleanup_cameras(stage)
            cleanup_replicator(stage)
            await kit_app.next_update_async()

            if not load_drone.LoadDrone(cfg).run():
                raise RuntimeError("Failed to load asset")
            for _ in range(10):
                await kit_app.next_update_async()

            pos_cam = position_cameras.PositionCameras(cfg)
            cameras, _, _ = pos_cam.run()
            if not cameras:
                raise RuntimeError("Failed to position cameras")
            for _ in range(10):
                await kit_app.next_update_async()

            animate.Animate(cfg).run()
            for _ in range(20):
                await kit_app.next_update_async()

            render_cfg = cfg["render"]
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            await render.render_multi_camera_async(
                num_cameras=render_cfg["num_cameras"],
                num_frames=render_cfg["num_frames"],
                output_dir=output_dir,
                stage_fps=cfg.get("fps", 30.0),
                drone_prim_path=cfg["drone"]["prim_path"],
                render_passes=render_cfg.get("passes", {}),
                resolution=pos_cam.resolution,
                asset_type=cfg["drone"].get("type"),
                mask_config=render_cfg.get("mask", {}),
            )

            (Path(output_dir) / ".render_complete").touch()
            print(f"[SMART] {job_id}: SUCCESS")
            succeeded += 1

        except Exception as e:
            print(f"[SMART] {job_id}: FAILED - {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print(f"\n{'-' * 60}")
    print("Batch Execution Complete")
    print(f"Total: {len(combos)}")
    print(f"Succeeded: {succeeded}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"{'-' * 60}")

    simulation_app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to batch manifest JSON")
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        print(f"Manifest not found: {args.manifest}")
        sys.exit(1)

    asyncio.ensure_future(run_smart_batch(args.manifest))

    while simulation_app.is_running():
        simulation_app.update()
