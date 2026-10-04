"""Isaac Sim object animation; detects asset type (drone, bird) and animates accordingly."""

import omni.usd
import omni.timeline
import carb.settings
from pxr import UsdSkel, Usd
from utils.config_utils import resolve_fps_from_config, resolve_timeline_from_config
from .flight_path import FlightPath, resolve_root_under_world, ensure_mover_wrapper


class Animate:
    """Flight-path animation, optional skeletal loop."""

    TYPE_DRONE = "drone"
    TYPE_BIRD = "bird"
    TYPE_UNKNOWN = "unknown"

    # asset-type detection from file path
    BIRD_KEYWORDS = ["bird", "eagle", "hawk", "crow", "raven", "sparrow", "pigeon", "owl", "falcon"]
    DRONE_KEYWORDS = ["drone", "quadcopter", "uav", "copter", "multirotor"]

    def __init__(self, cfg):
        self.cfg = cfg
        self.drone_cfg = cfg.get("drone", {})
        self.verbose = cfg.get("execution", {}).get("verbose", False)
        self.stage_fps = resolve_fps_from_config(cfg)
        self.asset_type = self.drone_cfg.get("asset_type")  # explicit type, else detected

        self.skel_animation_range = None
        self.animation_speed = float(self.drone_cfg.get("animation_speed_factor", 1.0))
        if self.verbose:
            print(f"[ANIMATE] Animation speed factor: {self.animation_speed}")
            if self.animation_speed != 1.0:
                 print(f"[ANIMATE] Applying non-default animation speed: {self.animation_speed}")

    def run(self):
        """Animate object per type; returns asset prim or None."""
        if self.verbose:
            print("[ANIMATE] Starting animation setup...")
            if self.asset_type:
                print(f"[ANIMATE] Using explicit asset type: {self.asset_type}")

        prim_path = self.drone_cfg.get("prim_path", "/World/Drone")
        if not prim_path:
            print("[ANIMATE] ERROR: No prim_path specified in config")
            return None

        usd_path = self.drone_cfg.get("usd_path", "")
        wrap_with_mover = bool(self.drone_cfg.get("wrap_with_mover", True))
        mover_prim_path = self.drone_cfg.get("mover_prim_path", "/World/DroneMover")

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print("[ANIMATE] Error: No active stage found")
            return None

        self._set_stage_fps(stage)

        obj_prim = stage.GetPrimAtPath(prim_path)
        if not obj_prim.IsValid():
            print(f"[ANIMATE] Error: No prim found at {prim_path}")
            return None

        if not self.asset_type:
            self.asset_type = self._detect_asset_type(stage, obj_prim, usd_path)
            if self.verbose:
                print(f"[ANIMATE] Detected asset type: {self.asset_type}")

        # top-level root (direct child of /World)
        obj_root = resolve_root_under_world(obj_prim)

        # mover wrapper for flight path
        prim_to_animate = obj_root
        if wrap_with_mover:
            if obj_root.IsInstanceable() or obj_root.IsInstanceProxy():
                if self.verbose:
                    print("[ANIMATE] Root prim is instanceable/proxy; skipping wrapper and animating root directly.")
            else:
                prim_to_animate = ensure_mover_wrapper(stage, obj_root, mover_prim_path, self.verbose)

        # ParentPrims moved asset under mover; refresh handle so Usd.PrimRange
        # scopes SkelAnimation search to asset subtree
        if prim_to_animate != obj_root:
            new_path = prim_to_animate.GetPath().AppendChild(obj_root.GetPath().name)
            refreshed = stage.GetPrimAtPath(new_path)
            if refreshed.IsValid():
                obj_prim = refreshed
                if self.verbose:
                    print(f"[ANIMATE] Refreshed asset prim path: {new_path}")

        success = False
        if self.asset_type == self.TYPE_BIRD:
            success = self._animate_bird(stage, prim_to_animate, obj_prim)
        else:
            # drone: flight path only
            success = self._animate_drone(prim_to_animate)

        if success and self.verbose:
            print("[ANIMATE] Animation setup complete")

        return obj_prim

    def _detect_asset_type(self, stage, prim, usd_path):
        """Asset type from path keywords, else SkelAnimation presence (bird), else drone."""
        usd_path_lower = usd_path.lower()

        for keyword in self.BIRD_KEYWORDS:
            if keyword in usd_path_lower:
                return self.TYPE_BIRD

        for keyword in self.DRONE_KEYWORDS:
            if keyword in usd_path_lower:
                return self.TYPE_DRONE

        # skeletal animation implies creature
        has_skel_animation = self._find_skel_animation(stage, prim)
        if has_skel_animation:
            return self.TYPE_BIRD

        return self.TYPE_DRONE

    def _find_skel_animation(self, stage, prim):
        """True if prim subtree has SkelAnimation."""
        for p in Usd.PrimRange(prim):
            if p.IsA(UsdSkel.Animation):
                return True
        return False

    def _get_skel_animation_range(self, stage, root_prim=None):
        """(min_time, max_time) over SkelAnimation time samples, or (None, None).

        root_prim scopes search to its subtree.
        """
        min_time = float('inf')
        max_time = float('-inf')
        found_animation = False

        prim_iter = Usd.PrimRange(root_prim) if root_prim else stage.Traverse()
        for prim in prim_iter:
            if prim.IsA(UsdSkel.Animation):
                if self.verbose:
                    print(f"[ANIMATE] Found SkelAnimation: {prim.GetPath()}")

                for attr_name in ['translations', 'rotations', 'scales', 'blendShapeWeights']:
                    attr = prim.GetAttribute(attr_name)
                    if attr and attr.HasValue():
                        time_samples = attr.GetTimeSamples()
                        if time_samples:
                            found_animation = True
                            min_time = min(min_time, min(time_samples))
                            max_time = max(max_time, max(time_samples))
                            if self.verbose:
                                print(f"[ANIMATE]   - {attr_name}: {len(time_samples)} keyframes, "
                                      f"range [{min(time_samples):.1f} - {max(time_samples):.1f}]")

        if found_animation:
            return min_time, max_time
        return None, None

    def _animate_bird(self, stage, prim_to_animate, obj_prim=None):
        """Bird: linear flight path on prim_to_animate, wing-flap SkelAnimation looped.

        obj_prim scopes SkelAnimation search to asset. Returns success.
        """
        if self.verbose:
            print("[ANIMATE] Setting up bird animation...")

        # wing-flap cycle range, scoped to asset
        skel_anim_start, skel_anim_end = self._get_skel_animation_range(stage, root_prim=obj_prim)

        if skel_anim_start is not None and skel_anim_end is not None:
            self.skel_animation_range = (skel_anim_start, skel_anim_end)
            if self.verbose:
                print(f"[ANIMATE] Detected skeletal animation range: {skel_anim_start} - {skel_anim_end} frames")
        else:
            self.skel_animation_range = None
            if self.verbose:
                print("[ANIMATE] No skeletal animation found")

        # timeline from render.num_frames
        tl = resolve_timeline_from_config(self.cfg)
        start_frame = tl["start_frame"]
        middle_frame = tl["middle_frame"]
        end_frame = tl["end_frame"]

        self._setup_timeline_for_flight(start_frame, end_frame)

        # loop skeletal animation, scoped to asset
        if self.skel_animation_range:
            self._setup_skel_animation_loop(stage, skel_anim_start, skel_anim_end, start_frame, end_frame,
                                            root_prim=obj_prim)

        flight_path_helper = FlightPath(self.cfg, verbose=self.verbose)
        flight_path = flight_path_helper.calculate_path()

        success = flight_path_helper.animate_prim(
            prim_to_animate,
            flight_path=flight_path,
            start_frame=start_frame,
            middle_frame=middle_frame,
            end_frame=end_frame
        )

        if success and self.verbose:
            print("[ANIMATE] Bird flight path animation applied")

        return success

    def _setup_timeline_for_flight(self, start_frame, end_frame):
        """Timeline spans flight, no looping."""
        try:
            timeline = omni.timeline.get_timeline_interface()
            if not timeline:
                print("[ANIMATE] Warning: Could not get timeline interface")
                return

            start_seconds = start_frame / self.stage_fps
            end_seconds = end_frame / self.stage_fps

            timeline.set_start_time(start_seconds)
            timeline.set_end_time(end_seconds)
            timeline.set_current_time(start_seconds)

            timeline.set_looping(False)

            if self.verbose:
                duration = end_seconds - start_seconds
                print("[ANIMATE] Timeline configured for flight path:")
                print(f"[ANIMATE]   - Range: {start_frame} - {end_frame} frames")
                print(f"[ANIMATE]   - Duration: {duration:.2f}s")
                print("[ANIMATE]   - Looping: Disabled (flight path is linear)")

        except Exception as e:
            print(f"[ANIMATE] Warning: Failed to configure timeline: {e}")

    def _setup_skel_animation_loop(self, stage, skel_start, skel_end, flight_start, flight_end,
                                    root_prim=None):
        """Re-time SkelAnimation keyframes to repeat over flight [flight_start, flight_end].

        root_prim limits which SkelAnimations are modified.
        """
        try:
            skel_duration = skel_end - skel_start
            if skel_duration <= 0:
                return

            # speed > 1.0 shortens cycle
            skel_duration_scaled = skel_duration / self.animation_speed

            flight_duration = flight_end - flight_start
            num_loops = int(flight_duration / skel_duration_scaled) + 1

            if self.verbose:
                print("[ANIMATE] Setting up skeletal animation loop:")
                print(f"[ANIMATE]   - Original cycle: {skel_duration} frames")
                print(f"[ANIMATE]   - Scaled cycle: {skel_duration_scaled:.2f} frames (speed: x{self.animation_speed})")
                print(f"[ANIMATE]   - Flight duration: {flight_duration} frames")
                print(f"[ANIMATE]   - Loops needed: {num_loops}")

            prim_iter = Usd.PrimRange(root_prim) if root_prim else stage.Traverse()
            for prim in prim_iter:
                if prim.IsA(UsdSkel.Animation):
                    self._extend_skel_animation_keyframes(
                        prim, skel_start, skel_end, flight_start, flight_end, num_loops, skel_duration_scaled
                    )

        except Exception as e:
            print(f"[ANIMATE] Warning: Failed to set up skeletal animation loop: {e}")

    def _extend_skel_animation_keyframes(self, skel_anim_prim, skel_start, skel_end,
                                          flight_start, flight_end, num_loops, skel_duration_scaled):
        """Repeat cycle [skel_start, skel_end] num_loops times from flight_start, clipped at flight_end."""
        for attr_name in ['translations', 'rotations', 'scales', 'blendShapeWeights']:
            attr = skel_anim_prim.GetAttribute(attr_name)
            if not attr or not attr.HasValue():
                continue

            time_samples = attr.GetTimeSamples()
            if not time_samples:
                continue

            # keyframes relative to skel_start
            original_keyframes = {}
            for t in time_samples:
                if skel_start <= t <= skel_end:
                    original_keyframes[t - skel_start] = attr.Get(t)

            if not original_keyframes:
                if self.verbose:
                     print(f"[ANIMATE] Warning: No keyframes found in range {skel_start}-{skel_end} for {attr.GetName()}")
                continue

            # clear, then rewrite starting at flight_start
            offset = flight_start - skel_start
            attr.Clear()
            # handle may go stale after Clear (same as flight_path.py)
            attr = skel_anim_prim.GetAttribute(attr_name)

            for loop_idx in range(num_loops):
                loop_offset = loop_idx * skel_duration_scaled + offset
                for rel_time, value in original_keyframes.items():
                    rel_time_scaled = rel_time / self.animation_speed
                    
                    new_time = rel_time_scaled + skel_start + loop_offset
                    if new_time <= flight_end:
                        attr.Set(value, new_time)

        if self.verbose:
            print(f"[ANIMATE]   Extended keyframes for {skel_anim_prim.GetPath()}")

    def _animate_drone(self, prim_to_animate):
        """Flight path only. Returns success."""
        if self.verbose:
            print("[ANIMATE] Setting up drone animation...")

        flight_path_helper = FlightPath(self.cfg, verbose=self.verbose)
        flight_path = flight_path_helper.calculate_path()

        if self.verbose:
            print(f"[ANIMATE] Flight path: {flight_path}")

        tl = resolve_timeline_from_config(self.cfg)
        return flight_path_helper.animate_prim(
            prim_to_animate,
            flight_path=flight_path,
            start_frame=tl["start_frame"],
            middle_frame=tl["middle_frame"],
            end_frame=tl["end_frame"],
        )

    def _set_stage_fps(self, stage):
        """Stage, timeline and kit FPS from config."""
        try:
            stage.SetTimeCodesPerSecond(self.stage_fps)
            stage.SetFramesPerSecond(self.stage_fps)

            timeline = omni.timeline.get_timeline_interface()
            if timeline:
                timeline.set_time_codes_per_second(self.stage_fps)
                timeline.set_ticks_per_frame(1)

            kit_settings = carb.settings.get_settings()
            if kit_settings:
                kit_settings.set("/app/player/useFixedTimeStepping", True)
                kit_settings.set("/app/stage/timeCodesPerSecond", self.stage_fps)

            if self.verbose:
                print(f"[ANIMATE] Set stage and timeline FPS to {self.stage_fps}")
        except Exception as e:
            print(f"[ANIMATE] Warning: Failed to set FPS: {e}")
