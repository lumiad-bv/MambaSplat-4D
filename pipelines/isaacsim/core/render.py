#!/usr/bin/env python3

"""Isaac Sim multi-camera render: RGB frames, binary object masks, camera metadata."""

import ast
from concurrent.futures import ThreadPoolExecutor
import json
import os
import shutil
from pathlib import Path

import carb.settings

import numpy as np
import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from omni.syntheticdata import SyntheticData
from PIL import Image

from utils.render_utils import (
    reshape_to_matrix as _reshape_to_matrix,
    build_camera_metadata as _build_camera_metadata,
    compute_world_bbox_3d as _compute_world_bbox_3d,
    project_point_to_screen as _project_point_to_screen,
    find_drone_prim as _find_drone_prim,
    safe_timeline_call as _safe_timeline_call,
    create_camera_api_helper as _create_camera_api_helper,
)


def _parse_color_tuple(color_str):
    """BasicWriter color string to int tuple, or None."""
    try:
        value = ast.literal_eval(color_str)
    except (SyntaxError, ValueError):
        return None

    if isinstance(value, (list, tuple)):
        try:
            return tuple(int(c) for c in value)
        except (TypeError, ValueError):
            return None
    return None


def _pack_color(color_tuple, num_channels=4):
    """RGB(A) tuple to uint32 for fast isin."""
    r = color_tuple[0] if len(color_tuple) > 0 else 0
    g = color_tuple[1] if len(color_tuple) > 1 else 0
    b = color_tuple[2] if len(color_tuple) > 2 else 0
    a = color_tuple[3] if len(color_tuple) > 3 else 0
    if num_channels <= 3:
        return np.uint32((r << 16) | (g << 8) | b)
    return np.uint32((r << 24) | (g << 16) | (b << 8) | a)


def _pack_pixels(img):
    """HxWxC uint8 image to HxW uint32 for fast isin."""
    if img.ndim == 2:
        return img.astype(np.uint32)
    if img.shape[2] >= 4:
        return (img[..., 0].astype(np.uint32) << 24 |
                img[..., 1].astype(np.uint32) << 16 |
                img[..., 2].astype(np.uint32) << 8 |
                img[..., 3].astype(np.uint32))
    return (img[..., 0].astype(np.uint32) << 16 |
            img[..., 1].astype(np.uint32) << 8 |
            img[..., 2].astype(np.uint32))


def _build_flying_object_colors(id2label, valid_object_paths, asset_type=None):
    """Inst-seg colors of flying object: prim path under valid_object_paths, or class label matching asset_type."""
    flying_colors = set()

    for color_str, prim_info in id2label.items():
        color_tuple = _parse_color_tuple(color_str)
        if color_tuple is None:
            continue

        if isinstance(prim_info, dict):
            # label format: {"class": "drone", ...}
            label = prim_info.get("class", "")
            if asset_type and label and label.lower().startswith(asset_type.lower()):
                flying_colors.add(color_tuple)
        else:
            # prim-path format: "/World/DroneMover/Drone/mesh"
            prim_path_str = str(prim_info)
            for prefix in valid_object_paths:
                if prim_path_str.startswith(prefix):
                    flying_colors.add(color_tuple)
                    break

    return flying_colors


def _process_mask_frame(inst_png_path_str, mask_dir_str, frame,
                        flying_colors_packed,
                        rgb_png_path_str=None, alpha_threshold=0):
    """One frame to binary mask.

    Primary: alpha of RGBA render (backgroundZeroAlpha, sub-pixel coverage);
    also composites RGB over white (sim1 / LGM need white background).
    Fallback: inst-seg color match.
    Returns {visible, center?, bbox?}.
    """
    mask = None

    # alpha mask
    if rgb_png_path_str is not None and os.path.exists(rgb_png_path_str):
        try:
            rgba_img = Image.open(rgb_png_path_str)
            if rgba_img.mode == "RGBA":
                rgba = np.array(rgba_img)
                alpha = rgba[:, :, 3]
                mask = np.where(
                    alpha > alpha_threshold, 255, 0).astype(np.uint8)

                # composite over white
                alpha_f = alpha.astype(np.float32) / 255.0
                rgb = rgba[:, :, :3].astype(np.float32)
                composited = rgb + 255.0 * (1.0 - alpha_f[:, :, np.newaxis])
                composited = np.clip(composited, 0, 255).astype(np.uint8)
                Image.fromarray(composited).save(rgb_png_path_str)
        except Exception as exc:
            print(f"[RENDER] Warning: alpha mask / composite failed "
                  f"for frame {frame}: {exc}")

    # fallback: inst-seg
    if mask is None:
        img = np.array(Image.open(inst_png_path_str))
        packed = _pack_pixels(img)
        if len(flying_colors_packed) > 0:
            mask = np.isin(packed, flying_colors_packed).astype(np.uint8) * 255
        else:
            mask = np.zeros(packed.shape, dtype=np.uint8)

    mask_path = Path(mask_dir_str) / f"drone_mask_{frame:04d}.png"
    Image.fromarray(mask).save(str(mask_path))

    if mask.any():
        ys, xs = np.where(mask > 0)
        return {
            "visible": True,
            "center": [float(np.mean(xs)), float(np.mean(ys))],
            "bbox": {
                "x_min": float(np.min(xs)),
                "y_min": float(np.min(ys)),
                "x_max": float(np.max(xs)),
                "y_max": float(np.max(ys)),
            },
        }
    return {"visible": False}


def _cleanup_inst_seg_files(camera_output_dirs):
    """Delete BasicWriter instance_* files.

    After mask extraction, and in RGB-only pass (inst seg force-enabled there
    to keep SyntheticData OmniGraph stable between passes).
    """
    for camera_output_dir in camera_output_dirs:
        for file_path in Path(camera_output_dir).glob("instance_*"):
            try:
                file_path.unlink()
            except Exception:
                pass


def _reconcile_metadata_from_disk_masks(
    camera_names, mask_output_dirs, num_frames, frame_records
):
    """Fill frame_records[*].cameras[*].{visible, drone_center_2d, bbox_2d} from drone_mask_*.png on disk.

    Idempotent; empty masks keep defaults. Keeps JSON consistent with PNGs if
    _process_mask_frame's in-memory result failed. Read-only; preserves `depth`.
    """
    for cam_idx, mdir in enumerate(mask_output_dirs):
        if mdir is None or cam_idx >= len(camera_names):
            continue
        mdir_path = Path(mdir)
        for frame in range(num_frames):
            if frame >= len(frame_records):
                break
            cams = frame_records[frame].get("cameras", [])
            if cam_idx >= len(cams):
                continue
            mp = mdir_path / f"drone_mask_{frame:04d}.png"
            if not mp.is_file():
                continue
            try:
                arr = np.asarray(Image.open(mp).convert("L"))
            except Exception:
                continue
            fg = arr > 0
            if not fg.any():
                continue  # empty mask, keep defaults
            ys, xs = np.where(fg)
            rec = cams[cam_idx]
            rec["visible"] = True
            rec["drone_center_2d"] = [float(xs.mean()), float(ys.mean())]
            rec["bbox_2d"] = {
                "x_min": float(xs.min()),
                "y_min": float(ys.min()),
                "x_max": float(xs.max()),
                "y_max": float(ys.max()),
            }


def _extract_instance_seg_masks(
    camera_names, camera_output_dirs, mask_output_dirs,
    valid_object_paths, asset_type, frame_records,
    alpha_threshold=0,
):
    """Binary masks per camera from RGBA alpha + inst-seg PNGs.

    Probes inst-seg mapping JSONs for object colors (UNLABELLED fallback after
    asset reload), threadpools _process_mask_frame, folds visibility/bbox into
    frame_records, deletes inst-seg files.
    """
    camera_tasks = []
    camera_packed_colors = {}

    for idx, camera_name in enumerate(camera_names):
        camera_output_dir = camera_output_dirs[idx]
        mask_dir = mask_output_dirs[idx]
        if mask_dir is None:
            continue

        inst_png_paths = sorted(Path(camera_output_dir).glob(
            "instance_segmentation_*.png"))
        if not inst_png_paths:
            print(f"[RENDER] Warning: No instance segmentation files "
                  f"found for {camera_name}; skipping mask extraction")
            continue

        flying_colors = set()
        id2label_len = 0
        n = len(inst_png_paths)
        probe_indices = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
        id2label = {}
        for pi in probe_indices:
            probe_path = inst_png_paths[pi]
            try:
                probe_frame = int(probe_path.stem.split("_")[-1])
            except (IndexError, ValueError):
                probe_frame = 0
            inst_map_json = probe_path.with_name(
                f"instance_segmentation_mapping_{probe_frame:04d}.json")
            if not inst_map_json.exists():
                continue
            try:
                with open(inst_map_json, "r") as f:
                    id2label = json.load(f)
                id2label_len = max(id2label_len, len(id2label))
                flying_colors |= _build_flying_object_colors(
                    id2label, valid_object_paths, asset_type=asset_type)
            except Exception as exc:
                print(f"[RENDER] Warning: Failed to parse id2label for "
                      f"{camera_name} frame {probe_frame}: {exc}")
            if flying_colors:
                break

        if flying_colors:
            print(f"[RENDER] {camera_name}: Found {len(flying_colors)} "
                  "flying object colors")
        else:
            # UNLABELLED fallback: SyntheticData may drop labels after asset
            # reload; on blank stage UNLABELLED covers only the asset
            unlabelled_color = (0, 0, 0, 255)
            for color_str, _prim_info in id2label.items():
                if _parse_color_tuple(color_str) == unlabelled_color:
                    flying_colors.add(unlabelled_color)
                    break
            if flying_colors:
                print(f"[RENDER] {camera_name}: Using UNLABELLED fallback "
                      "for mask (SyntheticData lost semantic labels)")
            else:
                print(f"[RENDER] {camera_name}: No flying object colors "
                      f"found (checked {id2label_len} entries, valid paths: "
                      f"{valid_object_paths})")

        first_img = np.array(Image.open(inst_png_paths[0]))
        num_ch = first_img.shape[2] if first_img.ndim == 3 else 1
        if flying_colors:
            packed = np.array(
                [_pack_color(c, num_ch) for c in flying_colors],
                dtype=np.uint32)
        else:
            packed = np.array([], dtype=np.uint32)
        camera_packed_colors[idx] = packed

        frame_items = []
        for p in inst_png_paths:
            try:
                f = int(p.stem.split("_")[-1])
                # rgb_*.png still beside inst-seg PNG, not yet moved to rgb/
                rgb_path = os.path.join(
                    camera_output_dir, f"rgb_{f:04d}.png")
                frame_items.append((str(p), str(mask_dir), f, rgb_path))
            except (IndexError, ValueError):
                print(f"[RENDER] Warning: Could not parse frame index "
                      f"from {p.name}; skipping")
        camera_tasks.append((idx, frame_items))

    # (inst_path, mask_dir, frame, packed_colors, cam_idx, rgb_path)
    all_tasks = [
        (inst_path, mdir, frame, camera_packed_colors[cam_idx],
         cam_idx, rgb_path)
        for cam_idx, frame_items in camera_tasks
        for inst_path, mdir, frame, rgb_path in frame_items
    ]

    if all_tasks:
        num_workers = min(os.cpu_count() or 1, len(all_tasks))
        print(f"[RENDER] Processing {len(all_tasks)} mask frames "
              f"with {num_workers} workers...")
        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(
                        _process_mask_frame, t[0], t[1], t[2], t[3],
                        rgb_png_path_str=t[5],
                        alpha_threshold=alpha_threshold)
                    for t in all_tasks
                ]
                results = [f.result() for f in futures]
        except Exception as exc:
            print(f"[RENDER] Thread pool failed ({exc}), "
                  "falling back to sequential...")
            results = [
                _process_mask_frame(
                    t[0], t[1], t[2], t[3],
                    rgb_png_path_str=t[5],
                    alpha_threshold=alpha_threshold)
                for t in all_tasks
            ]

        for task, result in zip(all_tasks, results):
            frame, cam_idx = task[2], task[4]
            if (frame < len(frame_records)
                    and cam_idx < len(frame_records[frame]["cameras"])):
                if result["visible"]:
                    cam_rec = frame_records[frame]["cameras"][cam_idx]
                    cam_rec["drone_center_2d"] = result["center"]
                    cam_rec["visible"] = True
                    cam_rec["bbox_2d"] = result["bbox"]
            elif frame >= len(frame_records):
                print(f"[RENDER] Note: Mask frame {frame:04d} for "
                      f"{camera_names[cam_idx]} has no matching frame "
                      "record entry; skipping metadata update")

    _cleanup_inst_seg_files(camera_output_dirs)


async def render_multi_camera_async(
    num_cameras=5,
    num_frames=120,
    output_dir=None,
    stage_fps=30.0,
    drone_prim_path=None,
    render_passes=None,
    rt_subframes=32,
    resolution=None,
    asset_type=None,
    mask_config=None,
):
    """Render all cameras; save RGB frames, binary masks, camera metadata.

    Args:
        stage_fps: drives stage, timeline and capture cadence
        drone_prim_path: flying object prim (3D bbox + mask filter)
        render_passes: {rgb: bool, instance_segmentation: bool}
        rt_subframes: path-tracing subframes per capture; lower = faster, noisier
        resolution: (width, height), must match camera intrinsics
        asset_type: class label selecting object inst-seg colors
        mask_config: {"method": "instance_seg", "alpha_threshold": 0}
    """
    if resolution is None:
        resolution = (1920, 1080)
        print(f"[RENDER] WARNING: No resolution provided, using default {resolution}")
    
    if render_passes is None:
        render_passes = {}
    if mask_config is None:
        mask_config = {"method": "instance_seg"}
    mask_method = mask_config.get("method", "alpha")

    print("[RENDER] *** ASYNC TASK STARTED *** (This confirms the task is executing)")
    
    if output_dir is None:
        raise ValueError("output_dir is required")

    os.makedirs(output_dir, exist_ok=True)
    
    print("[RENDER] Starting multi-camera render")
    print(f"[RENDER] Number of cameras: {num_cameras}")
    print(f"[RENDER] Number of frames: {num_frames}")
    print(f"[RENDER] Resolution: {resolution[0]}x{resolution[1]}")
    print(f"[RENDER] Output directory: {output_dir}")
    
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[RENDER] Error: No active stage found")
        return

    try:
        original_stage_timecodes = float(stage.GetTimeCodesPerSecond())
    except Exception:
        original_stage_timecodes = None
    try:
        original_stage_frames = float(stage.GetFramesPerSecond())
    except Exception:
        original_stage_frames = None

    try:
        stage_fps = float(stage_fps)
    except (TypeError, ValueError):
        stage_fps = 30.0
    if stage_fps <= 0.0:
        stage_fps = 30.0

    try:
        stage.SetTimeCodesPerSecond(stage_fps)
        stage.SetFramesPerSecond(stage_fps)
    except Exception as exc:
        print(f"[RENDER] Warning: Failed to apply stage FPS {stage_fps}: {exc}")

    try:
        applied_stage_fps = float(stage.GetTimeCodesPerSecond() or stage_fps)
    except Exception:
        applied_stage_fps = stage_fps
    if applied_stage_fps <= 0.0:
        applied_stage_fps = stage_fps if stage_fps > 0.0 else 30.0
    if abs(applied_stage_fps - stage_fps) > 1e-3:
        print(
            f"[RENDER] Warning: Stage reports {applied_stage_fps:.2f} FPS after applying {stage_fps:.2f}. Using reported value."
        )
    stage_fps = applied_stage_fps
    frame_dt = 1.0 / stage_fps

    synthetic_data = SyntheticData.Get()
    previous_semantic_filter = None
    try:
        previous_semantic_filter = synthetic_data.get_instance_mapping_semantic_filter()
    except Exception:
        pass


    mask_output_dirs = []
    camera_output_dirs = []
    rgb_dirs = []
    camera_names = []
    camera_prims = []
    camera_param_annots = []
    camera_helpers = []
    camera_metadata_cache = {}
    frame_records = []
    render_products = []
    writers = []

    # object prim for 3D bbox
    drone_prim = _find_drone_prim(stage, preferred_path=drone_prim_path)
    drone_prim_path = str(drone_prim.GetPath()) if drone_prim and drone_prim.IsValid() else drone_prim_path

    # mover prim (parent) for mask filter
    mover_prim_path = None
    if drone_prim and drone_prim.IsValid():
        parent = drone_prim.GetParent()
        if parent and parent.IsValid() and "mover" in str(parent.GetPath()).lower():
            mover_prim_path = str(parent.GetPath())

    # fallback mover paths
    if not mover_prim_path:
        for path in ["/World/DroneMover", "/World/ObjectMover"]:
            mover_candidate = stage.GetPrimAtPath(path)
            if mover_candidate and mover_candidate.IsValid():
                mover_prim_path = path
                break

    # prim-path prefixes for mask filter
    valid_object_paths = []
    if mover_prim_path:
        valid_object_paths.append(mover_prim_path)
    if drone_prim_path:
        valid_object_paths.append(drone_prim_path)

    if drone_prim_path:
        print(f"[RENDER] Found flying object prim at: {drone_prim_path}")
    if mover_prim_path:
        print(f"[RENDER] Found mover prim at: {mover_prim_path}")
    if not valid_object_paths:
        print("[RENDER] Warning: Could not find drone/object prim. 3D bbox tracking and mask creation will be disabled.")

    timeline = omni.timeline.get_timeline_interface()
    kit_settings = carb.settings.get_settings()
    previous_use_fixed_timestep = None
    previous_stage_timecodes = None
    previous_bg_zero_alpha = {}
    if kit_settings:
        try:
            previous_use_fixed_timestep = kit_settings.get("/app/player/useFixedTimeStepping")
        except Exception:
            previous_use_fixed_timestep = None
        try:
            previous_stage_timecodes = kit_settings.get("/app/stage/timeCodesPerSecond")
        except Exception:
            previous_stage_timecodes = None
        # save backgroundZeroAlpha state; restored in finally
        for _key in [
            "/rtx/post/backgroundZeroAlpha/enabled",
            "/rtx/post/backgroundZeroAlpha/backgroundComposite",
            "/rtx/post/backgroundZeroAlpha/outputAlphaInComposite",
            "/rtx/post/backgroundZeroAlpha/blackBackgroundInComposite",
        ]:
            try:
                previous_bg_zero_alpha[_key] = kit_settings.get(_key)
            except Exception:
                pass

    previous_timeline_fps = _safe_timeline_call(timeline, "get_time_codes_per_seconds")
    previous_time = _safe_timeline_call(timeline, "get_current_time") or 0.0
    previous_end_time = _safe_timeline_call(timeline, "get_end_time")
    was_playing = bool(_safe_timeline_call(timeline, "is_playing"))

    # stop before FPS change so it applies
    _safe_timeline_call(timeline, "stop")

    if kit_settings:
        try:
            kit_settings.set("/app/player/useFixedTimeStepping", True)
        except Exception as exc:
            print(f"[RENDER] Warning: Failed to enable fixed time stepping: {exc}")
        try:
            kit_settings.set("/app/stage/timeCodesPerSecond", stage_fps)
        except Exception:
            pass
        # keyframe motion blur ghosts moving asset
        try:
            kit_settings.set("/rtx/post/motionblur/enabled", False)
        except Exception:
            pass
        # mask pass only: backgroundZeroAlpha writes sub-pixel coverage to
        # alpha, catches thin rods / distant silhouettes inst seg misses
        if mask_method == "instance_seg":
            try:
                kit_settings.set("/rtx/post/backgroundZeroAlpha/enabled", True)
                kit_settings.set("/rtx/post/backgroundZeroAlpha/backgroundComposite", False)
                kit_settings.set("/rtx/post/backgroundZeroAlpha/outputAlphaInComposite", True)
                kit_settings.set("/rtx/post/backgroundZeroAlpha/blackBackgroundInComposite", True)
                print("[RENDER] Enabled backgroundZeroAlpha for mask pass "
                      "(alpha ∪ inst-seg union)")
            except Exception as exc:
                print(f"[RENDER] Warning: Failed to enable backgroundZeroAlpha: {exc}")
        else:
            try:
                kit_settings.set("/rtx/post/backgroundZeroAlpha/enabled", False)
            except Exception:
                pass

    # timeline FPS, retry once
    timeline_fps = stage_fps
    if timeline:
        try:
            timeline.set_time_codes_per_second(stage_fps)
        except Exception as exc:
            print(f"[RENDER] Warning: Failed to set timeline FPS {stage_fps}: {exc}")
        
        # ticks per frame = 1
        _safe_timeline_call(timeline, "set_ticks_per_frame", 1)
        
        fetched = _safe_timeline_call(timeline, "get_time_codes_per_seconds")
        if fetched:
            try:
                fetched_fps = float(fetched)
                if abs(fetched_fps - stage_fps) > 1e-3:
                    print(f"[RENDER] Timeline FPS mismatch detected ({fetched_fps:.2f} vs {stage_fps:.2f}), retrying...")
                    _safe_timeline_call(timeline, "set_time_codes_per_second", stage_fps)
                    fetched = _safe_timeline_call(timeline, "get_time_codes_per_seconds")
                    if fetched:
                        fetched_fps = float(fetched)
                
                timeline_fps = fetched_fps
                if abs(timeline_fps - stage_fps) > 1e-3:
                    print(
                        f"[RENDER] Warning: Timeline FPS {timeline_fps:.2f} differs from stage FPS {stage_fps:.2f}"
                    )
                    print(f"[RENDER] Using stage FPS {stage_fps:.2f} for frame timing regardless")
                    timeline_fps = stage_fps
            except (TypeError, ValueError):
                timeline_fps = stage_fps

    print(f"[RENDER] Stage FPS: {stage_fps:.2f} (Δt={frame_dt:.4f}s)")

    # critical for RGB/mask sync: only explicit step_async() captures,
    # once per call for all render products
    rep.orchestrator.set_capture_on_play(False)

    kit_app = omni.kit.app.get_app()

    _safe_timeline_call(timeline, "stop")
    _safe_timeline_call(timeline, "set_current_time", 0.0)

    try:
        # render product + writer per camera
        for i in range(num_cameras):
            camera_name = f"cam_{i+1:02d}"
            camera_path = f"/World/{camera_name}"

            camera_prim = stage.GetPrimAtPath(camera_path)
            if not camera_prim.IsValid():
                print(f"[RENDER] Warning: Camera {camera_path} not found, skipping")
                continue

            print(f"[RENDER] Setting up render product for {camera_name} at {camera_path}")

            rp = rep.create.render_product(camera_path, resolution)
            render_products.append(rp)

            writer = rep.writers.get("BasicWriter")
            camera_output_dir = os.path.join(output_dir, camera_name)
            os.makedirs(camera_output_dir, exist_ok=True)
            rgb_dir = os.path.join(camera_output_dir, "rgb")
            mask_dir = os.path.join(camera_output_dir, "mask")
            enable_rgb = render_passes.get("rgb", True)
            enable_inst_seg = render_passes.get("instance_segmentation", True)
            # mask pass needs inst-seg
            if mask_method == "instance_seg":
                enable_inst_seg = True

            print(f"[RENDER] {camera_name} configuration:")
            print(f"[RENDER]   RGB: {enable_rgb}")
            print(f"[RENDER]   Instance Segmentation: {enable_inst_seg}")

            if enable_rgb:
                os.makedirs(rgb_dir, exist_ok=True)
            if enable_inst_seg:
                os.makedirs(mask_dir, exist_ok=True)

            writer.initialize(
                output_dir=camera_output_dir,
                rgb=enable_rgb,
                instance_segmentation=enable_inst_seg,
            )

            writer.attach(rp)
            writers.append(writer)
            
            cam_params_annot = rep.AnnotatorRegistry.get_annotator("camera_params")
            cam_params_annot.attach(rp)

            if enable_inst_seg:
                mask_output_dirs.append(mask_dir)
            else:
                mask_output_dirs.append(None)

            camera_output_dirs.append(camera_output_dir)
            
            if enable_rgb:
                rgb_dirs.append(rgb_dir)
            else:
                rgb_dirs.append(None)

            camera_names.append(camera_name)
            camera_prims.append(camera_prim)
            camera_param_annots.append(cam_params_annot)

            rp_path_attr = getattr(rp, "path", None)
            helper = _create_camera_api_helper(
                camera_path,
                camera_name,
                str(rp_path_attr) if rp_path_attr is not None else None,
            )
            camera_helpers.append(helper)

            print(f"[RENDER] {camera_name} output: {camera_output_dir}")
            print(f"[RENDER] {camera_name} drone masks: {mask_dir}")

        if not render_products:
            print("[RENDER] Error: No valid cameras found")
            return

        print(f"[RENDER] Successfully set up {len(render_products)} camera render products")

        # let GPU finish deferred render-product allocs; else timeline.commit() may segfault
        if kit_app and hasattr(kit_app, "next_update_async"):
            for _ in range(3):
                await kit_app.next_update_async()

        # keep timeline stopped; step_async advances it
        _safe_timeline_call(timeline, "stop")
        _safe_timeline_call(timeline, "set_current_time", 0.0)
        
        # end time covers all frames
        desired_end_time = max(previous_end_time or 0.0, num_frames * frame_dt)
        if desired_end_time <= 0.0:
            desired_end_time = max(num_frames * frame_dt, 1.0)
        _safe_timeline_call(timeline, "set_end_time", desired_end_time)
        _safe_timeline_call(timeline, "commit")
        # no play(): avoids capture at t=0 before loop

        # detach/re-attach resets BasicWriter frame counters so frame 0 is
        # first controlled capture
        print("[RENDER] Resetting writer frame counters to ensure proper synchronization...")
        for writer in writers:
            writer.detach()
        
        # let detach complete
        if kit_app and hasattr(kit_app, "next_update_async"):
            await kit_app.next_update_async()
        
        for idx, (writer, rp) in enumerate(zip(writers, render_products)):
            writer.attach(rp)
        
        print("[RENDER] Writers reset and re-attached, frame counters start at 0...")

        print(f"[RENDER] Starting frame capture (frames 0000-{num_frames-1:04d}, total {num_frames} frames)...")
        frames_captured = 0

        cam_params_annot = rep.AnnotatorRegistry.get_annotator("camera_params")
        for rp in render_products:
            cam_params_annot.attach(rp)

        # per frame: set timeline, update stage, 3D bbox, step_async(delta_time=0.0)
        # captures all cameras without advancing time; masks post-processed after
        try:
            for frame in range(0, num_frames):
                current_time = frame * frame_dt
                _safe_timeline_call(timeline, "set_current_time", current_time)
                
                # apply timeline to stage
                if kit_app and hasattr(kit_app, "next_update_async"):
                    await kit_app.next_update_async()
                
                # world-space 3D bbox
                drone_bbox_3d = None
                if drone_prim and drone_prim.IsValid():
                    drone_bbox_3d = _compute_world_bbox_3d(drone_prim, seconds=current_time)
                
                # capture without advancing time; BasicWriter writes RGB + inst-seg
                await rep.orchestrator.step_async(rt_subframes=rt_subframes, delta_time=0.0, pause_timeline=True)
                frames_captured += 1

                frame_record = {
                    "frame_index": frame,
                    "time_seconds": float(current_time),
                    "drone_position_3d": drone_bbox_3d["center"] if drone_bbox_3d else None,
                    "cameras": [],
                }

                for idx in range(len(camera_names)):
                    width = height = None
                    camera_entry = {
                        "name": camera_names[idx],
                        "drone_center_2d": None,
                        "visible": False,
                        "depth": None,
                        "bbox_2d": None,
                    }

                    camera_params = None
                    if idx < len(camera_param_annots):
                        try:
                            camera_params = camera_param_annots[idx].get_data()
                        except Exception:
                            pass

                    if camera_params:
                        try:
                            resolution = camera_params["renderProductResolution"]
                            width = int(resolution[0])
                            height = int(resolution[1])

                            if camera_names[idx] not in camera_metadata_cache:
                                metadata = _build_camera_metadata(
                                    camera_names[idx], camera_prims[idx], render_products[idx], camera_params
                                )
                                if metadata:
                                    camera_metadata_cache[camera_names[idx]] = metadata
                            
                            # depth from projected center
                            if drone_bbox_3d and drone_bbox_3d.get("center") is not None:
                                try:
                                    view_matrix = _reshape_to_matrix(camera_params["cameraViewTransform"])
                                    projection_matrix = _reshape_to_matrix(camera_params["cameraProjection"])
                                    
                                    center_proj = _project_point_to_screen(
                                        drone_bbox_3d["center"],
                                        view_matrix,
                                        projection_matrix,
                                        width,
                                        height,
                                    )
                                    if center_proj and center_proj["in_front"]:
                                        camera_entry["depth"] = float(center_proj["depth"])
                                except Exception as exc:
                                    print(f"[RENDER] Warning: Failed to compute depth for {camera_names[idx]} frame {frame}: {exc}")
                        except Exception:
                            pass

                    frame_record["cameras"].append(camera_entry)

                frame_records.append(frame_record)

                if frame % 10 == 0 or frame == 0:
                    print(f"[RENDER] Captured frame {frame}/{num_frames-1}")
                    if drone_bbox_3d:
                        print(f"[RENDER]   Drone 3D bbox center: {drone_bbox_3d['center']}")
        finally:
            _safe_timeline_call(timeline, "stop")
            if previous_end_time is not None:
                _safe_timeline_call(timeline, "set_end_time", previous_end_time)
            _safe_timeline_call(timeline, "set_current_time", previous_time)
            _safe_timeline_call(timeline, "commit")
            if timeline and previous_timeline_fps is not None:
                try:
                    timeline.set_time_codes_per_second(previous_timeline_fps)
                except Exception:
                    pass
            if was_playing:
                _safe_timeline_call(timeline, "play")
            if original_stage_timecodes is not None:
                try:
                    stage.SetTimeCodesPerSecond(original_stage_timecodes)
                except Exception:
                    pass
            if original_stage_frames is not None:
                try:
                    stage.SetFramesPerSecond(original_stage_frames)
                except Exception:
                    pass
            if kit_settings and previous_stage_timecodes is not None:
                try:
                    kit_settings.set("/app/stage/timeCodesPerSecond", previous_stage_timecodes)
                except Exception:
                    pass
            if kit_settings and previous_use_fixed_timestep is not None:
                try:
                    kit_settings.set("/app/player/useFixedTimeStepping", previous_use_fixed_timestep)
                except Exception:
                    pass
            # restore backgroundZeroAlpha
            if kit_settings and previous_bg_zero_alpha:
                for _key, _prev in previous_bg_zero_alpha.items():
                    if _prev is not None:
                        try:
                            kit_settings.set(_key, _prev)
                        except Exception:
                            pass

        print(f"[RENDER] Frame capture complete ({frames_captured} frames)")

        # flush writers; files must be on disk
        # drain the writer queue before detaching: on a cold start the last
        # capture is otherwise dropped
        print("[RENDER] Waiting for file writes to complete...")
        if kit_app and hasattr(kit_app, "next_update_async"):
            for _ in range(3):
                await kit_app.next_update_async()
        await rep.orchestrator.wait_until_complete_async()
        for writer in writers:
            try:
                writer.detach()
            except Exception:
                pass
        writers.clear()  # avoid double-detach in finally
        await rep.orchestrator.wait_until_complete_async()

        # masks: blank stage, asset is only geometry; alpha (sub-pixel coverage)
        # primary, inst-seg fallback
        alpha_threshold = int(mask_config.get("alpha_threshold", 0))
        print(f"[RENDER] Post-processing masks "
              f"(alpha>{alpha_threshold}, inst-seg fallback)...")
        _extract_instance_seg_masks(
            camera_names, camera_output_dirs, mask_output_dirs,
            valid_object_paths, asset_type, frame_records,
            alpha_threshold=alpha_threshold)
        # backstop: reconcile JSON with mask PNGs on disk
        _reconcile_metadata_from_disk_masks(
            camera_names, mask_output_dirs, num_frames, frame_records,
        )

        print("[RENDER] Mask post-processing complete")

        # drop extra frames from BasicWriter pipeline delay
        for idx in range(len(camera_output_dirs)):
            cam_dir = camera_output_dirs[idx]
            for pattern in ["rgb/rgb_*.png", "mask/drone_mask_*.png",
                            "rgb_*.png"]:
                for p in Path(cam_dir).glob(pattern):
                    try:
                        fnum = int(p.stem.split("_")[-1])
                        if fnum >= num_frames:
                            p.unlink()
                    except (ValueError, IndexError):
                        pass

        # placeholder metadata for cameras without params
        for idx, name in enumerate(camera_names):
            if name not in camera_metadata_cache:
                rp_path = None
                if idx < len(render_products):
                    rp_candidate = getattr(render_products[idx], "path", None)
                    rp_path = str(rp_candidate) if rp_candidate is not None else None
                prim_path = None
                if idx < len(camera_prims):
                    prim = camera_prims[idx]
                    prim_path = str(prim.GetPath()) if prim and prim.IsValid() else None

                camera_metadata_cache[name] = {
                    "name": name,
                    "camera_prim_path": prim_path,
                    "render_product_path": rp_path,
                    "resolution": None,
                    "intrinsics": None,
                    "extrinsics": None,
                    "projection_matrix": None,
                    "meters_per_scene_unit": None,
                    "note": "Camera parameters not available; check camera configuration.",
                }

        observation_payload = {
            "description": "Drone world pose and per-camera projections exported from render.py",
            "metadata": {
                "num_frames": len(frame_records),
                "stage_fps": stage_fps,
                "timeline_fps": timeline_fps,
                "frame_dt": frame_dt,
                "drone_prim_path": drone_prim_path,
                "cameras_recorded": camera_names,
            },
            "cameras": list(camera_metadata_cache.values()),
            "frames": frame_records,
        }

        if frame_records:
            observation_payload["metadata"]["time_span_seconds"] = {
                "start": frame_records[0]["time_seconds"],
                "end": frame_records[-1]["time_seconds"],
            }
        else:
            observation_payload["metadata"]["time_span_seconds"] = {"start": 0.0, "end": 0.0}

        observation_path = os.path.join(output_dir, "drone_camera_observations.json")
        try:
            with open(observation_path, "w", encoding="utf-8") as json_file:
                json.dump(observation_payload, json_file, indent=2)
            print(f"[RENDER] Drone/camera observation JSON saved: {observation_path}")
        except Exception as exc:
            print(f"[RENDER] Warning: Failed to write observation JSON: {exc}")

        # move rgb_* into rgb/
        for cam_dir, rgb_dir in zip(camera_output_dirs, rgb_dirs):
            try:
                for entry in os.listdir(cam_dir):
                    src_path = os.path.join(cam_dir, entry)
                    if not os.path.isfile(src_path):
                        continue
                    if entry.startswith("rgb_") and rgb_dir:
                        shutil.move(src_path, os.path.join(rgb_dir, entry))
            except Exception:
                pass

        print(f"[RENDER] Rendering complete! Output saved to: {output_dir}")
    finally:
        # free render products (GPU leak across batch jobs); writers already cleared on success
        for writer in writers:
            try:
                writer.detach()
            except Exception:
                pass
        for annot in camera_param_annots:
            try:
                annot.detach()
            except Exception:
                pass
        for helper in camera_helpers:
            if helper is None:
                continue
            try:
                helper.destroy()
            except Exception:
                pass
        for rp in render_products:
            try:
                rp.destroy()
            except Exception:
                pass
        try:
            await rep.orchestrator.wait_until_complete_async()
        except Exception:
            pass
        if previous_semantic_filter is not None:
            try:
                synthetic_data.set_instance_mapping_semantic_filter(previous_semantic_filter)
            except Exception:
                pass
