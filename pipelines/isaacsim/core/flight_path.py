"""Flight path computation and translate-keyframe animation for any prim."""

import omni.usd
import omni.kit.commands
from pxr import UsdGeom, Gf, Sdf
from math import sqrt, atan2, degrees
from utils.config_utils import resolve_timeline_from_config


def _compute_nose_rotation(dx, dy, dz):
    """RotateXYZ Euler (deg) aligning +X with direction. Z-up: yaw=atan2(dy,dx) about Z, pitch=atan2(dz,horiz) about local Y, roll=0."""
    horiz = sqrt(dx * dx + dy * dy)
    yaw_deg = degrees(atan2(dy, dx))
    pitch_deg = degrees(atan2(dz, horiz)) if horiz > 1e-9 else (90.0 if dz > 0 else -90.0)
    return Gf.Vec3d(0.0, -pitch_deg, yaw_deg)


class FlightPath:
    """Compute and animate flight paths for prims."""

    def __init__(self, cfg, verbose=False):
        """cfg: config dict with 'drone' section."""
        self.cfg = cfg
        self.drone_cfg = cfg.get("drone", {})
        self.verbose = verbose

    def calculate_path(self, waypoint=None, flight_distance=None, flight_direction=None):
        """Start/end points of path centred on waypoint. None args fall back to config. Returns dict: start, waypoint, end, total_distance."""
        if waypoint is None:
            waypoint = tuple(self.drone_cfg.get("waypoint", [0.0, 0.0, 0.0]))
        if flight_distance is None:
            flight_distance = float(self.drone_cfg.get("flight_distance", 10.0))
        if flight_direction is None:
            flight_direction = tuple(self.drone_cfg.get("flight_direction", [1.0, 0.0, 0.0]))

        wx, wy, wz = waypoint

        # normalize
        dx, dy, dz = flight_direction
        length = sqrt(dx*dx + dy*dy + dz*dz)
        if length > 0:
            dx, dy, dz = dx/length, dy/length, dz/length
        else:
            dx, dy, dz = 1.0, 0.0, 0.0

        half_distance = flight_distance / 2.0

        start_point = (
            wx - dx * half_distance,
            wy - dy * half_distance,
            wz - dz * half_distance
        )

        end_point = (
            wx + dx * half_distance,
            wy + dy * half_distance,
            wz + dz * half_distance
        )

        return {
            "start": start_point,
            "waypoint": (wx, wy, wz),
            "end": end_point,
            "total_distance": flight_distance
        }

    def animate_prim(self, prim, flight_path=None, start_frame=None,
                     middle_frame=None, end_frame=None):
        """Translate keyframes along flight_path. None args fall back to calculate_path()/config. Returns success bool."""
        try:
            if not prim or not prim.IsValid():
                print("[FLIGHT_PATH] animate_prim received invalid prim; aborting.")
                return False

            if flight_path is None:
                flight_path = self.calculate_path()

            # timeline from render.num_frames
            if start_frame is None or middle_frame is None or end_frame is None:
                tl = resolve_timeline_from_config(self.cfg)
                if start_frame is None:
                    start_frame = tl["start_frame"]
                if middle_frame is None:
                    middle_frame = tl["middle_frame"]
                if end_frame is None:
                    end_frame = tl["end_frame"]

            prim = self._ensure_xformable(prim)
            if prim is None:
                return False

            xformable = UsdGeom.Xformable(prim)
            if not xformable:
                print("[FLIGHT_PATH] Could not obtain Xformable for prim; abort.")
                return False

            translate_op = self._get_or_create_translate_op(xformable)
            if translate_op is None:
                print("[FLIGHT_PATH] Failed to acquire/create translate op; abort.")
                return False

            # clear stale keyframes from previous runs
            attr = translate_op.GetAttr()
            if attr and attr.GetTimeSamples():
                attr.Clear()
                # Clear drops samples, keeps op definition; re-acquire
                translate_op = self._get_or_create_translate_op(xformable)

            # frame numbers as time codes
            translate_op.Set(Gf.Vec3d(*flight_path["start"]), start_frame)
            translate_op.Set(Gf.Vec3d(*flight_path["waypoint"]), middle_frame)
            translate_op.Set(Gf.Vec3d(*flight_path["end"]), end_frame)

            if self.verbose:
                print(f"[FLIGHT_PATH] Set keyframes: start={start_frame}, mid={middle_frame}, end={end_frame}")
                print(f"[FLIGHT_PATH] Path: {flight_path['start']} -> {flight_path['waypoint']} -> {flight_path['end']}")

            nose_pointing = self.drone_cfg.get("nose_pointing", True)
            if nose_pointing and flight_path:
                self._apply_nose_rotation(xformable, flight_path)

            return True
        except Exception as e:
            print(f"[FLIGHT_PATH] Failed to create animation: {e}")
            return False

    def _ensure_xformable(self, prim):
        """Return Xformable prim; wrap in parent Xform if needed."""
        if UsdGeom.Xformable(prim):
            return prim

        if prim.GetTypeName() not in ("Xform", "Scope", "Mesh", "Cube", "Cylinder", "Sphere"):
            if self.verbose:
                print("[FLIGHT_PATH] Prim not naturally Xformable; creating temporary parent Xform.")

            stage = prim.GetStage()
            temp_parent_path = prim.GetPath().GetParentPath().AppendChild(
                prim.GetPath().name + "_AnimXform"
            )

            if not stage.GetPrimAtPath(temp_parent_path).IsValid():
                stage.DefinePrim(temp_parent_path, "Xform")
                try:
                    omni.kit.commands.execute(
                        "ParentPrims",
                        parent_path=str(temp_parent_path),
                        child_paths=[str(prim.GetPath())],
                        keep_world_transform=True,
                    )
                    return stage.GetPrimAtPath(temp_parent_path)
                except Exception as e:
                    print(f"[FLIGHT_PATH] Failed to create temp animation parent: {e}")
                    return None
            else:
                return stage.GetPrimAtPath(temp_parent_path)

        return prim

    def _get_or_create_translate_op(self, xformable):
        """Existing or new translate op."""
        translate_ops = xformable.GetOrderedXformOps()
        for op in translate_ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                return op

        try:
            return xformable.AddTranslateOp()
        except Exception:
            return xformable.GetTranslateOp()

    def _apply_nose_rotation(self, xformable, flight_path):
        """Static RotateXYZ op so +X faces flight direction. Order [translate, rotateXYZ]: USD rotates locally, then translates."""
        sx, sy, sz = flight_path["start"]
        ex, ey, ez = flight_path["end"]
        dx, dy, dz = ex - sx, ey - sy, ez - sz
        length = sqrt(dx * dx + dy * dy + dz * dz)
        if length < 1e-9:
            return
        dx, dy, dz = dx / length, dy / length, dz / length

        rotation = _compute_nose_rotation(dx, dy, dz)

        rotate_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                rotate_op = op
                break
        if rotate_op is None:
            rotate_op = xformable.AddRotateXYZOp()

        # clear stale samples (batch re-runs)
        attr = rotate_op.GetAttr()
        if attr and attr.GetTimeSamples():
            attr.Clear()
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                    rotate_op = op
                    break

        rotate_op.Set(rotation)

        if self.verbose:
            print(f"[FLIGHT_PATH] Nose-pointing rotation: yaw={rotation[2]:.1f} deg, "
                  f"pitch={rotation[1]:.1f} deg")


def resolve_root_under_world(prim):
    """Ancestor of prim that is direct child of /World."""
    p = prim
    while p:
        parent = p.GetParent()
        if not parent:
            return p
        if parent.GetPath() == Sdf.Path("/World") or parent.GetPath() == Sdf.Path.absoluteRootPath:
            return p
        p = parent
    return prim


def ensure_mover_wrapper(stage, prim_root, mover_path, verbose=False):
    """Wrap prim_root under mover Xform at mover_path. Returns mover prim, or prim_root on failure."""
    mover_prim = stage.GetPrimAtPath(mover_path)
    if not mover_prim.IsValid():
        mover_prim = stage.DefinePrim(mover_path, "Xform")

    if prim_root.GetParent() == mover_prim:
        return mover_prim

    # reparent, keep world transform
    try:
        omni.kit.commands.execute(
            "ParentPrims",
            parent_path=str(mover_prim.GetPath()),
            child_paths=[str(prim_root.GetPath())],
            keep_world_transform=True,
        )
        new_child_path = mover_prim.GetPath().AppendChild(prim_root.GetPath().name)
        new_child = stage.GetPrimAtPath(new_child_path)
        if not new_child.IsValid():
            if verbose:
                print("[FLIGHT_PATH] Warning: parenting reported success but child prim not found under mover; falling back to root animation")
            return prim_root
    except Exception as e:
        if verbose:
            print(f"[FLIGHT_PATH] Failed to wrap prim with mover: {e}. Falling back to root prim animation.")
        return prim_root

    return mover_prim
