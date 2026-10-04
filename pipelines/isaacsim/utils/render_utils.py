"""Isaac Sim render helpers: matrices, intrinsics, bboxes, projection."""


import numpy as np
from pxr import Gf, Usd, UsdGeom

try:
    from isaacsim.sensors.camera import Camera as IsaacSimCamera
except ModuleNotFoundError:
    IsaacSimCamera = None


def reshape_to_matrix(matrix_values):
    """Flat 16 values to 4x4 row-major matrix."""
    array = np.array(matrix_values, dtype=np.float64).reshape(-1)
    if array.size != 16:
        raise ValueError(f"Expected 16 values to reshape into 4x4 matrix, got {array.size}")
    matrix = array.reshape(4, 4)

    # Replicator/USD give column-major; transpose if translation sits in last row
    translation_column_norm = np.linalg.norm(matrix[:3, 3])
    translation_row_norm = np.linalg.norm(matrix[3, :3])
    if translation_column_norm < translation_row_norm:
        matrix = matrix.T

    return matrix


def matrix_to_list(matrix: np.ndarray) -> list[list[float]]:
    """Nested float list."""
    return matrix.astype(np.float64).tolist()


def extract_camera_pose(view_matrix: np.ndarray):
    """camera_to_world, position, quaternion (w, x, y, z) from view matrix."""
    camera_to_world = np.linalg.inv(view_matrix)
    transform = Gf.Transform()
    transform.SetMatrix(Gf.Matrix4d(camera_to_world.tolist()))
    translation = transform.GetTranslation()
    rotation = transform.GetRotation().GetQuat()

    position = [float(translation[0]), float(translation[1]), float(translation[2])]
    quaternion = [
        float(rotation.GetReal()),
        float(rotation.GetImaginary()[0]),
        float(rotation.GetImaginary()[1]),
        float(rotation.GetImaginary()[2]),
    ]
    return camera_to_world, position, quaternion


def compute_camera_intrinsics(camera_params: dict) -> dict:
    """Intrinsics from camera annotator params."""
    resolution = camera_params["renderProductResolution"]
    width = int(resolution[0])
    height = int(resolution[1])

    aperture = np.asarray(camera_params["cameraAperture"], dtype=np.float64)
    aperture_offset = np.asarray(camera_params["cameraApertureOffset"], dtype=np.float64)
    focal_length = float(camera_params["cameraFocalLength"])

    if width == 0 or aperture[0] == 0.0:
        raise ValueError("Invalid camera aperture or resolution for intrinsic computation")

    pixel_size = aperture[0] / width
    fx = focal_length / pixel_size
    fy = focal_length / pixel_size
    cx = width / 2.0 + aperture_offset[0]
    cy = height / 2.0 + aperture_offset[1]

    intrinsic_matrix = [
        [float(fx), 0.0, float(cx)],
        [0.0, float(fy), float(cy)],
        [0.0, 0.0, 1.0],
    ]

    return {
        "matrix": intrinsic_matrix,
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "focal_length_mm": focal_length,
        "aperture": aperture.astype(float).tolist(),
        "aperture_offset": aperture_offset.astype(float).tolist(),
    }


def build_camera_metadata(camera_name, camera_prim, render_product, camera_params):
    """Static camera metadata: intrinsics + extrinsics. None on failure."""
    try:
        view_matrix = reshape_to_matrix(camera_params["cameraViewTransform"])
        projection_matrix = reshape_to_matrix(camera_params["cameraProjection"])
        intrinsics = compute_camera_intrinsics(camera_params)
        camera_to_world, position, quaternion = extract_camera_pose(view_matrix)
    except Exception as exc:
        print(f"[RENDER] Warning: Unable to build metadata for {camera_name}: {exc}")
        return None

    render_product_path = getattr(render_product, "path", None)
    if render_product_path is not None:
        render_product_path = str(render_product_path)

    camera_path = str(camera_prim.GetPath()) if camera_prim and camera_prim.IsValid() else None
    resolution = camera_params["renderProductResolution"]
    width = int(resolution[0])
    height = int(resolution[1])

    metadata = {
        "name": camera_name,
        "camera_prim_path": camera_path,
        "render_product_path": render_product_path,
        "resolution": {"width": width, "height": height},
        "intrinsics": intrinsics,
        "extrinsics": {
            "world_to_camera_matrix": matrix_to_list(view_matrix),
            "camera_to_world_matrix": matrix_to_list(camera_to_world),
            "position_world": position,
            "orientation_quat_wxyz": quaternion,
        },
        "projection_matrix": matrix_to_list(projection_matrix),
        "meters_per_scene_unit": float(camera_params["metersPerSceneUnit"]),
    }
    return metadata


def compute_world_bbox_3d(prim, seconds=None):
    """World AABB of prim at `seconds` (None = default time). Returns dict center/extents/min/max/corners_world, or None."""
    if not prim or not prim.IsValid():
        return None
    
    try:
        stage = prim.GetStage()
        tcps = float(stage.GetTimeCodesPerSecond() or 30.0)
        time_code = Usd.TimeCode.Default() if seconds is None else Usd.TimeCode(seconds * tcps)
        
        purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
        bbox_cache = UsdGeom.BBoxCache(time_code, purposes, useExtentsHint=True)
        
        bound = bbox_cache.ComputeWorldBound(prim)
        bound_range = bound.ComputeAlignedBox()
        
        min_point = bound_range.GetMin()
        max_point = bound_range.GetMax()
        
        center = [
            (min_point[0] + max_point[0]) / 2.0,
            (min_point[1] + max_point[1]) / 2.0,
            (min_point[2] + max_point[2]) / 2.0,
        ]
        
        extents = [
            max_point[0] - min_point[0],
            max_point[1] - min_point[1],
            max_point[2] - min_point[2],
        ]
        
        corners = [
            [min_point[0], min_point[1], min_point[2]],
            [min_point[0], min_point[1], max_point[2]],
            [min_point[0], max_point[1], min_point[2]],
            [min_point[0], max_point[1], max_point[2]],
            [max_point[0], min_point[1], min_point[2]],
            [max_point[0], min_point[1], max_point[2]],
            [max_point[0], max_point[1], min_point[2]],
            [max_point[0], max_point[1], max_point[2]],
        ]
        
        return {
            "center": center,
            "extents": extents,
            "min": [float(min_point[0]), float(min_point[1]), float(min_point[2])],
            "max": [float(max_point[0]), float(max_point[1]), float(max_point[2])],
            "corners_world": corners,
        }
    except Exception as exc:
        print(f"[RENDER] Warning: Failed to compute world bbox for {prim.GetPath()}: {exc}")
        return None


def project_point_to_screen(world_point, view_matrix, projection_matrix, screen_width, screen_height):
    """World point to screen via 4x4 view (world-to-camera) and projection. Returns dict pixel/ndc/depth/in_front, or None."""
    try:
        world_point_h = np.array([world_point[0], world_point[1], world_point[2], 1.0], dtype=np.float64)

        camera_point = view_matrix @ world_point_h

        # OpenGL convention: camera looks down -Z
        in_front = bool(camera_point[2] < 0)

        clip_point = projection_matrix @ camera_point

        if abs(clip_point[3]) < 1e-8:
            return None

        ndc = clip_point[:3] / clip_point[3]

        # NDC [-1,1] to pixels, Y flipped
        pixel_x = (ndc[0] + 1.0) * screen_width * 0.5
        pixel_y = (1.0 - ndc[1]) * screen_height * 0.5

        depth = float(-camera_point[2])

        return {
            "pixel": [float(pixel_x), float(pixel_y)],
            "ndc": [float(ndc[0]), float(ndc[1])],
            "depth": depth,
            "in_front": in_front,
        }
    except Exception:
        return None


def find_drone_prim(stage, preferred_path=None):
    """Asset prim: preferred_path, then common paths, then keyword search. None if absent."""
    if not stage:
        return None
    
    try:
        if preferred_path:
            prim = stage.GetPrimAtPath(preferred_path)
            if prim and prim.IsValid():
                return prim
        
        common_paths = [
            "/World/drone",
            "/World/Drone",
            "/World/bird",
            "/World/Bird",
            "/World/Eagle",
            "/World/Helicopter",
            "/World/helicopter",
            "/World/Airplane",
            "/World/airplane",
            "/drone",
            "/Drone",
        ]

        for path in common_paths:
            prim = stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                return prim

        keywords = ["drone", "bird", "eagle", "seeker", "copternode",
                     "helicopter", "heli", "airplane", "aircraft", "cessna", "cesna"]
        for prim in stage.Traverse():
            path_str = str(prim.GetPath()).lower()
            if any(kw in path_str for kw in keywords):
                return prim
        
        return None
    except Exception:
        return None


def safe_timeline_call(timeline, method_name, *args, **kwargs):
    """Call timeline method; None on failure."""
    try:
        method = getattr(timeline, method_name)
        return method(*args, **kwargs)
    except Exception:
        return None


def create_camera_api_helper(camera_path, camera_name, render_product_path=None):
    """isaacsim.sensors.camera.Camera helper, or None."""
    if IsaacSimCamera is None:
        return None

    try:
        return IsaacSimCamera(
            prim_path=camera_path,
            name=f"_render_helper_{camera_name}",
            render_product_path=render_product_path,
        )
    except Exception:
        return None
