"""Blank Isaac Sim stage lit by white dome light."""

import omni.usd
import omni.timeline
import carb.settings
from pxr import Gf, UsdGeom, Sdf

from utils.config_utils import resolve_fps_from_config


class LoadScene:
    """Blank white stage all assets render on."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.verbose = cfg.get("execution", {}).get("verbose", False)
        self.stage_fps = resolve_fps_from_config(cfg)

    def run(self):
        return self._create_blank_stage()

    def _create_blank_stage(self):
        """New stage: /World + white DomeLight only. No ground plane, no distant light."""
        print("[LOAD] Creating blank stage (no scene)")

        usd_context = omni.usd.get_context()
        stage = usd_context.get_stage()

        world_prim = stage.GetPrimAtPath("/World") if stage else None
        if world_prim and world_prim.IsValid():
            marker = world_prim.GetCustomData().get("source_scene_path")
            if marker == "__blank__":
                if self.verbose:
                    print("[LOAD] Blank stage already active, skipping recreate")
                self._set_stage_fps(stage)
                return world_prim

        usd_context.new_stage()
        stage = usd_context.get_stage()
        if stage is None:
            print("[LOAD] ERROR: Failed to create new stage")
            return None

        # Z-up (Isaac Sim convention)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        world_prim = stage.DefinePrim("/World", "Xform")
        world_prim.SetCustomDataByKey("source_scene_path", "__blank__")

        # untextured white dome: uniform background, no geometry
        dome = stage.DefinePrim("/World/DomeLight", "DomeLight")
        dome.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(1500.0)
        dome.CreateAttribute("inputs:exposure", Sdf.ValueTypeNames.Float).Set(0.0)
        dome.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(1.0, 1.0, 1.0)
        )
        dome.CreateAttribute("inputs:texture:file", Sdf.ValueTypeNames.Asset).Set("")

        self._set_stage_fps(stage)

        print("[LOAD] Blank stage created (white dome light background only)")
        return world_prim

    def _set_stage_fps(self, stage):
        if stage is None:
            return
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
                print(f"[LOAD] Set stage and timeline FPS to {self.stage_fps}")
        except Exception as exc:
            print(f"[LOAD] Warning: Failed to set FPS: {exc}")
