"""Load USD/FBX asset into stage; detect type; apply semantic labels for instance segmentation."""

import omni.usd
from isaacsim.core.utils.semantics import (  # type: ignore
    add_labels,
    upgrade_prim_semantics_to_labels,
)
from pxr import Usd, UsdGeom, Gf


TYPE_DRONE = "drone"
TYPE_BIRD = "bird"
TYPE_HELICOPTER = "helicopter"
TYPE_AIRPLANE = "airplane"
TYPE_UNKNOWN = "unknown"

# type detection from file path
BIRD_KEYWORDS = ["bird", "eagle", "hawk", "crow", "raven", "sparrow", "pigeon", "owl", "falcon"]
DRONE_KEYWORDS = ["drone", "quadcopter", "uav", "multirotor"]
HELICOPTER_KEYWORDS = ["helicopter", "heli", "huey", "little bird", "md-500", "ec-135", "ec 135", "md-902"]
AIRPLANE_KEYWORDS = ["airplane", "aircraft", "cessna", "cesna", "aerobatic"]


def _set_semantic_label(prim, label):
    """Set class label via Labels API."""
    if prim is None or not prim.IsValid():
        return False

    try:
        add_labels(prim, labels=[label], instance_name="class")
        return True
    except Exception:
        return False


def _detect_asset_type(usd_path):
    """TYPE_* constant from keywords in usd_path."""
    usd_path_lower = usd_path.lower()

    for keyword in BIRD_KEYWORDS:
        if keyword in usd_path_lower:
            return TYPE_BIRD

    for keyword in HELICOPTER_KEYWORDS:
        if keyword in usd_path_lower:
            return TYPE_HELICOPTER

    for keyword in AIRPLANE_KEYWORDS:
        if keyword in usd_path_lower:
            return TYPE_AIRPLANE

    for keyword in DRONE_KEYWORDS:
        if keyword in usd_path_lower:
            return TYPE_DRONE

    return TYPE_UNKNOWN


class LoadDrone:
    """Load USD/FBX asset into Isaac Sim stage."""

    def __init__(self, cfg):
        """cfg: config dict."""
        self.cfg = cfg
        self.drone_cfg = cfg.get("drone", {})
        self.verbose = cfg.get("execution", {}).get("verbose", False)
        self.last_semantics_applied = False
        self.asset_type = None
    
    def run(self):
        """Load asset; return prim or None."""
        usd_path = self.drone_cfg.get("usd_path")
        if not usd_path:
            print("[LOAD_OBJECT] ERROR: No usd_path specified in config")
            return None

        prim_path = self.drone_cfg.get("prim_path", "/World/Drone")

        # explicit type from asset_config.yaml, else auto-detect
        explicit_type = self.drone_cfg.get("type") or self.drone_cfg.get("asset_type")
        self.asset_type = explicit_type if explicit_type else _detect_asset_type(usd_path)

        if self.verbose:
            print(f"[LOAD_OBJECT] Loading asset from: {usd_path}")
            print(f"[LOAD_OBJECT] Target prim path: {prim_path}")
            print(f"[LOAD_OBJECT] Detected asset type: {self.asset_type}")

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print("[LOAD_OBJECT] ERROR: No active stage found")
            return None

        obj_prim = stage.DefinePrim(prim_path, "Xform")
        obj_prim.GetReferences().AddReference(usd_path)

        self._apply_config_transforms(obj_prim)

        if self.verbose:
            print(f"[LOAD_OBJECT] ✓ Asset loaded at: {prim_path}")

        try:
            self.last_semantics_applied = bool(self.apply_semantic_labels(prim_path))
            if self.verbose:
                if self.last_semantics_applied:
                    print(f"[LOAD_OBJECT] ✓ Applied semantic labels at {prim_path}")
                else:
                    print(f"[LOAD_OBJECT] WARNING: Failed to apply semantic labels at {prim_path}")
        except Exception as exc:
            self.last_semantics_applied = False
            print(f"[LOAD_OBJECT] WARNING: Exception while applying semantics: {exc}")

        return obj_prim

    def _apply_config_transforms(self, prim):
        """Translation, orientation, scale from config."""
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            return

        translation = self.drone_cfg.get("translation")
        if translation:
            try:
                xform_ops = xformable.GetOrderedXformOps()
                translate_op = None
                for op in xform_ops:
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        translate_op = op
                        break
                
                if not translate_op:
                    translate_op = xformable.AddTranslateOp()
                
                translate_op.Set(Gf.Vec3d(translation))
                if self.verbose:
                    print(f"[LOAD_OBJECT] Applied translation: {translation}")
            except Exception as e:
                print(f"[LOAD_OBJECT] WARNING: Failed to apply translation: {e}")

        # Euler degrees
        orientation = self.drone_cfg.get("orientation")
        if orientation:
            try:
                xform_ops = xformable.GetOrderedXformOps()
                rotate_op = None
                for op in xform_ops:
                    if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                        rotate_op = op
                        break
                
                if not rotate_op:
                    rotate_op = xformable.AddRotateXYZOp()
                
                rotate_op.Set(Gf.Vec3d(orientation))
                if self.verbose:
                    print(f"[LOAD_OBJECT] Applied orientation: {orientation}")
            except Exception as e:
                print(f"[LOAD_OBJECT] WARNING: Failed to apply orientation: {e}")

        scale = self.drone_cfg.get("scale")
        if scale:
            try:
                xform_ops = xformable.GetOrderedXformOps()
                scale_op = None
                for op in xform_ops:
                    if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                        scale_op = op
                        break
                
                if not scale_op:
                    scale_op = xformable.AddScaleOp()
                
                scale_op.Set(Gf.Vec3f(scale))
                if self.verbose:
                    print(f"[LOAD_OBJECT] Applied scale: {scale}")
            except Exception as e:
                print(f"[LOAD_OBJECT] WARNING: Failed to apply scale: {e}")

    def apply_semantic_labels(self, prim_path):
        """Label root and Imageable descendants by asset type. Returns success bool."""
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print("[LOAD_OBJECT] ERROR: No active stage found")
            return False

        label_success = False

        root_prim = stage.GetPrimAtPath(prim_path)
        if not root_prim or not root_prim.IsValid():
            print(f"[LOAD_OBJECT] WARNING: Prim not found at {prim_path}")
            return False

        try:
            upgrade_prim_semantics_to_labels(root_prim, include_descendants=True)
        except Exception as exc:
            if self.verbose:
                print(f"[LOAD_OBJECT] WARNING: Failed to upgrade semantics at {prim_path}: {exc}")

        labeled_prims = 0
        verbose_print_limit = 20

        base_label = self.asset_type if self.asset_type else TYPE_UNKNOWN

        if _set_semantic_label(root_prim, base_label):
            label_success = True
            labeled_prims += 1
            if self.verbose and labeled_prims <= verbose_print_limit:
                print(f"[LOAD_OBJECT] ✓ Labeled {prim_path} as '{base_label}'")

        for prim in Usd.PrimRange(root_prim):
            if prim == root_prim:
                continue
            if not prim.IsValid():
                continue

            if not prim.IsA(UsdGeom.Imageable):
                continue

            path_str = str(prim.GetPath()).lower()
            label = self._determine_part_label(path_str, base_label)

            if _set_semantic_label(prim, label):
                label_success = True
                labeled_prims += 1
                if self.verbose and labeled_prims <= verbose_print_limit:
                    print(f"[LOAD_OBJECT] ✓ Labeled {prim.GetPath()} as '{label}'")

        if self.verbose:
            if label_success:
                print(f"[LOAD_OBJECT] ✓ Applied semantics to {labeled_prims} prims under {prim_path}")
            else:
                print(f"[LOAD_OBJECT] WARNING: No semantics applied under {prim_path}")

        return label_success

    def _determine_part_label(self, path_str, base_label):
        """Part label from lowercase prim path and base_label."""
        if base_label == TYPE_DRONE:
            if "prop" in path_str:
                return "drone_prop"
            elif "body" in path_str or "hull" in path_str:
                return "drone_body"
            else:
                return "drone"
        elif base_label == TYPE_BIRD:
            if "wing" in path_str:
                return "bird_wing"
            elif "head" in path_str or "beak" in path_str:
                return "bird_head"
            elif "tail" in path_str:
                return "bird_tail"
            elif "body" in path_str or "torso" in path_str:
                return "bird_body"
            elif "leg" in path_str or "foot" in path_str or "claw" in path_str:
                return "bird_leg"
            else:
                return "bird"
        else:
            return base_label
