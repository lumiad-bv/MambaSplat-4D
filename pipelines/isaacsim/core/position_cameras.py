#!/usr/bin/env python3

"""Isaac Sim camera rig: ground circle, sphere or 4-view LGM layout, aimed at a center point."""

import numpy as np
from math import pi, cos, sin, sqrt, acos
from pathlib import Path
import yaml

import utils.config_utils as config_utils

from utils.config_utils import resolve_fps_from_config

from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera
import omni.usd
import omni.kit.commands
import omni.client
from pxr import UsdGeom
from isaacsim.core.utils.stage import get_stage_units


def _collect_search_roots():
    roots = []
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()
    roots.extend([script_dir, cwd])

    try:
        cfg_module_dir = Path(config_utils.__file__).resolve().parent
        roots.append(cfg_module_dir)
    except Exception:
        cfg_module_dir = None

    deduped = []
    seen = set()
    for root in filter(None, roots):
        lineage = [root] + list(root.parents)
        for candidate in lineage:
            key = str(candidate)
            if key not in seen:
                deduped.append(candidate)
                seen.add(key)
    return deduped


_SEARCH_ROOTS = _collect_search_roots()


def _resolve_existing_path(path_arg):
    path_candidate = Path(path_arg)

    if path_candidate.is_absolute():
        return path_candidate if path_candidate.exists() else None

    seen = set()
    for base_dir in _SEARCH_ROOTS:
        if base_dir is None:
            continue
        for candidate in (base_dir / path_candidate, base_dir / "configs" / path_candidate):
            try:
                resolved = candidate.resolve()
            except Exception:
                resolved = candidate
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            if resolved.exists():
                return resolved

    return None


def load_camera_intrinsics_from_yaml(yaml_path="cam_intrinsics.yaml"):
    """Intrinsics database from YAML; searches _SEARCH_ROOTS."""
    resolved_yaml = _resolve_existing_path(yaml_path)
    if resolved_yaml is None:
        raise FileNotFoundError(f"Camera intrinsics file not found: {Path(yaml_path)}")

    with open(resolved_yaml, 'r') as f:
        intrinsics_db = yaml.safe_load(f)
    
    return intrinsics_db


def get_camera_intrinsics(camera_type, intrinsics_yaml="cam_intrinsics.yaml"):
    """Intrinsics dict for camera_type, e.g. "advanced_pinhole", "fisheye"."""
    intrinsics_db = load_camera_intrinsics_from_yaml(intrinsics_yaml)
    
    if camera_type not in intrinsics_db:
        available = list(intrinsics_db.keys())
        raise ValueError(
            f"Camera type '{camera_type}' not found in {intrinsics_yaml}. "
            f"Available types: {', '.join(available)}"
        )
    
    return intrinsics_db[camera_type]


def get_scene_parameters():
    """Up axis and units of current stage."""
    stage = omni.usd.get_context().get_stage()
    
    if stage is None:
        print("ERROR: No stage is currently loaded!")
        return None
    
    up_axis_token = UsdGeom.GetStageUpAxis(stage)
    up_axis = str(up_axis_token)
    
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    stage_units = get_stage_units()
    
    if meters_per_unit == 1.0:
        units_description = "meters (m)"
        units_short = "m"
    elif meters_per_unit == 0.01:
        units_description = "centimeters (cm)"
        units_short = "cm"
    elif meters_per_unit == 0.001:
        units_description = "millimeters (mm)"
        units_short = "mm"
    elif meters_per_unit == 1000.0:
        units_description = "kilometers (km)"
        units_short = "km"
    else:
        units_description = f"custom ({meters_per_unit} meters per unit)"
        units_short = "custom"
    
    scene_params = {
        "up_axis": up_axis,
        "meters_per_unit": meters_per_unit,
        "stage_units": stage_units,
        "units_description": units_description,
        "units_short": units_short
    }
    
    return scene_params


def scale_K_to_resolution(intr_cfg, target_width, target_height):
    """Copy of intrinsics with K scaled to target resolution."""
    if not intr_cfg:
        return intr_cfg

    scaled = intr_cfg.copy()
    K = np.array(intr_cfg.get("camera_matrix", np.eye(3)), dtype=float)
    original_resolution = intr_cfg.get("original_resolution") or [target_width, target_height]
    orig_w = float(original_resolution[0]) if original_resolution[0] else float(target_width)
    orig_h = float(original_resolution[1]) if original_resolution[1] else float(target_height)

    sx = float(target_width) / orig_w if orig_w else 1.0
    sy = float(target_height) / orig_h if orig_h else 1.0

    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] *= sx
    K[1, 2] *= sy

    scaled["camera_matrix"] = K.tolist()
    scaled["original_resolution"] = [target_width, target_height]
    return scaled


def fibonacci_sphere_points(n, hemisphere="full"):
    """n Fibonacci-lattice points on unit sphere; hemisphere "full", "upper" (z >= 0) or "lower" (z <= 0)."""
    golden_ratio = (1 + sqrt(5)) / 2

    if hemisphere == "full":
        sample_n = n
    else:
        # oversample, then filter
        sample_n = n * 2

    points = []
    for i in range(sample_n):
        phi = acos(1 - 2 * (i + 0.5) / sample_n)
        theta = 2 * pi * i / golden_ratio
        x = sin(phi) * cos(theta)
        y = sin(phi) * sin(theta)
        z = cos(phi)

        if hemisphere == "upper" and z < 0:
            continue
        if hemisphere == "lower" and z > 0:
            continue

        points.append((x, y, z))
        if len(points) >= n:
            break

    return points


def look_at_quaternion(camera_pos, target_pos):
    """[w, x, y, z] quaternion looking from camera_pos at target_pos; ZYX Euler, roll = 0."""
    dx = target_pos[0] - camera_pos[0]
    dy = target_pos[1] - camera_pos[1]
    dz = target_pos[2] - camera_pos[2]

    horizontal_dist = np.sqrt(dx * dx + dy * dy)
    yaw = np.arctan2(dy, dx)
    pitch = -np.arctan2(dz, horizontal_dist)
    roll = 0.0

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


class PositionCameras:
    """Create and aim cameras: circle, sphere or LGM placement."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.cameras_cfg = cfg.get("cameras", {})
        self.verbose = cfg.get("execution", {}).get("verbose", False)
        self.cameras = []
        self.aim_point = None
        self.scene_params = None
        self.resolution = None
    
    def run(self):
        """Returns (cameras, aim_point, scene_params)."""
        if self.verbose:
            print("[CAMS] Starting camera setup...")

        auto_detect = self.cameras_cfg.get("auto_detect_scene", True)
        if auto_detect:
            if self.verbose:
                print("[CAMS] Detecting scene parameters...")
            self.scene_params = get_scene_parameters()

        cfg = self.cameras_cfg

        RESOLUTION = tuple(cfg.get("resolution", [512, 512]))

        # intrinsics first: original_resolution overrides default
        camera_type = cfg.get("camera_type", "advanced_pinhole")
        intrinsics_yaml = cfg.get("intrinsics_yaml", "cam_intrinsics.yaml")

        try:
            intrinsics_cfg = get_camera_intrinsics(camera_type, intrinsics_yaml)
            if self.verbose:
                print(f"[CAMS] Loaded '{camera_type}' intrinsics from {intrinsics_yaml}")
                print(f"[CAMS]   Description: {intrinsics_cfg.get('description', 'N/A')}")
            # K calibrated for original_resolution
            orig_res = intrinsics_cfg.get("original_resolution")
            if orig_res:
                RESOLUTION = tuple(orig_res)
                if self.verbose:
                    print(f"[CAMS]   Using intrinsics resolution: {RESOLUTION}")
        except (FileNotFoundError, ValueError) as e:
            print(f"[CAMS] WARNING: {e}")
            print("[CAMS] Using default intrinsics configuration")
            intrinsics_cfg = {}

        self.resolution = RESOLUTION
        if cfg.get("frequency") is not None:
            FREQUENCY = int(cfg.get("frequency"))
        else:
            FREQUENCY = max(1, int(round(resolve_fps_from_config(self.cfg))))

        focus_distance_override = cfg.get("focus_distance", None)
        if focus_distance_override is not None:
            intrinsics_cfg = intrinsics_cfg.copy()
            intrinsics_cfg["focus_distance"] = focus_distance_override

        stage_units = self.scene_params['stage_units'] if self.scene_params else 1.0
        try:
            world = World.instance()
            if world is None:
                world = World(stage_units_in_meters=stage_units)
        except:
            world = World(stage_units_in_meters=stage_units)

        if self.verbose:
            print(f"[CAMS] Using stage units: {stage_units} meters per unit")

        # LGM overrides
        lgm_cfg = self.cfg.get("lgm", {})
        if lgm_cfg.get("enabled", False):
            cfg["placement"] = "lgm"
            cfg["camera_type"] = "lgm_pinhole"
            lgm_res = list(lgm_cfg.get("resolution", [256, 256]))
            cfg["resolution"] = lgm_res
            RESOLUTION = tuple(lgm_res)
            self.resolution = RESOLUTION
            # initial load used pre-override camera_type
            try:
                intrinsics_cfg = get_camera_intrinsics("lgm_pinhole", intrinsics_yaml)
                # keep FOV at render resolution
                intrinsics_cfg = scale_K_to_resolution(
                    intrinsics_cfg, lgm_res[0], lgm_res[1])
                if self.verbose:
                    print(f"[CAMS] Reloaded 'lgm_pinhole' intrinsics "
                          f"(scaled to {lgm_res[0]}x{lgm_res[1]})")
            except (FileNotFoundError, ValueError) as e:
                print(f"[CAMS] WARNING: Failed to load lgm_pinhole intrinsics: {e}")
            if self.verbose:
                print(f"[CAMS] LGM preset active: 4 cameras, lgm_pinhole, "
                      f"{lgm_res[0]}x{lgm_res[1]}")

        placement = cfg.get("placement", "circle")
        if placement == "lgm":
            self._place_cameras_lgm(cfg, FREQUENCY, RESOLUTION)
        elif placement == "sphere":
            self._place_cameras_sphere(cfg, FREQUENCY, RESOLUTION)
        else:
            self._place_cameras_circle(cfg, FREQUENCY, RESOLUTION)

        initialize_cameras(self.cameras, self.scene_params, intrinsics_cfg, self.aim_point)

        if intrinsics_cfg:
            self._configure_intrinsics(intrinsics_cfg)

        if self.verbose:
            print(f"[CAMS] Created {len(self.cameras)} cameras ({placement} placement)")

        return self.cameras, self.aim_point, self.scene_params

    def _place_cameras_circle(self, cfg, frequency, resolution):
        """Circle on ground plane."""
        circle_cfg = cfg.get("circle", {})
        NUM_CAMERAS = max(1, min(8, int(circle_cfg.get("num_cameras", 5))))
        SIDE_METERS = float(circle_cfg.get("side_meters", 5.0))
        CAM_HEIGHT = float(circle_cfg.get("cam_height", 2.0))
        AIM_POINT_HEIGHT_OFFSET = float(circle_cfg.get("aim_point_height_offset", -5.0))
        ROTATION_OFFSET = float(circle_cfg.get("rotation_offset", -pi/2))
        POSITION_OFFSET_X = float(circle_cfg.get("position_offset_x", 1800.0))
        POSITION_OFFSET_Y = float(circle_cfg.get("position_offset_y", 8000.0))

        radius = SIDE_METERS / 2.0
        cam_height = CAM_HEIGHT
        aim_height = cam_height + AIM_POINT_HEIGHT_OFFSET

        if self.scene_params and self.scene_params['up_axis'] == 'Y':
            self.aim_point = np.array([POSITION_OFFSET_X, aim_height, POSITION_OFFSET_Y])
            if self.verbose:
                print("[CAMS] Using Y-up coordinate system")
        else:
            self.aim_point = np.array([POSITION_OFFSET_X, POSITION_OFFSET_Y, aim_height])
            if self.verbose:
                print("[CAMS] Using Z-up coordinate system")

        if self.scene_params and self.scene_params['meters_per_unit'] != 1.0:
            scale_factor = 1.0 / self.scene_params['meters_per_unit']
            radius *= scale_factor
            cam_height *= scale_factor
            self.aim_point *= scale_factor
            POSITION_OFFSET_X *= scale_factor
            POSITION_OFFSET_Y *= scale_factor

        for i in range(NUM_CAMERAS):
            ang = 2.0 * pi * i / NUM_CAMERAS + ROTATION_OFFSET

            if self.scene_params and self.scene_params['up_axis'] == 'Y':
                cam_pos = np.array([
                    radius * cos(ang) + POSITION_OFFSET_X,
                    cam_height,
                    radius * sin(ang) + POSITION_OFFSET_Y,
                ])
            else:
                cam_pos = np.array([
                    radius * cos(ang) + POSITION_OFFSET_X,
                    radius * sin(ang) + POSITION_OFFSET_Y,
                    cam_height,
                ])

            camera_name = f"cam_{i + 1:02d}"
            quaternion_wxyz = look_at_quaternion(cam_pos, self.aim_point)

            camera = Camera(
                prim_path=f"/World/{camera_name}",
                position=cam_pos,
                frequency=frequency,
                resolution=resolution,
                orientation=quaternion_wxyz,
            )
            self.cameras.append(camera)

            if self.verbose:
                distance_to_aim = np.linalg.norm(self.aim_point - cam_pos)
                print(f"[CAMS] {camera_name}: pos={cam_pos}, distance_to_aim={distance_to_aim:.2f}")

    def _place_cameras_sphere(self, cfg, frequency, resolution):
        """Fibonacci sphere, looking at center."""
        sphere_cfg = cfg.get("sphere", {})
        radius = float(sphere_cfg.get("radius", cfg.get("circle", {}).get("side_meters", 10.0)))
        center = list(sphere_cfg.get("center", [0.0, 0.0, 15.0]))
        num_cameras = int(sphere_cfg.get("num_cameras", cfg.get("num_cameras", 20)))
        hemisphere = sphere_cfg.get("hemisphere", "full")

        y_up = self.scene_params and self.scene_params['up_axis'] == 'Y'
        if self.verbose:
            axis_label = "Y-up" if y_up else "Z-up"
            print(f"[CAMS] Sphere placement: {num_cameras} cameras, radius={radius}, "
                  f"center={center}, hemisphere={hemisphere} ({axis_label})")

        if self.scene_params and self.scene_params['meters_per_unit'] != 1.0:
            scale_factor = 1.0 / self.scene_params['meters_per_unit']
            radius *= scale_factor
            center = [c * scale_factor for c in center]

        self.aim_point = np.array(center)

        # lattice pole is Z; Y-up stages swap Y and Z below
        points = fibonacci_sphere_points(num_cameras, hemisphere=hemisphere)

        for i, (px, py, pz) in enumerate(points):
            if y_up:
                cam_pos = np.array([
                    px * radius + center[0],
                    pz * radius + center[1],
                    py * radius + center[2],
                ])
            else:
                cam_pos = np.array([
                    px * radius + center[0],
                    py * radius + center[1],
                    pz * radius + center[2],
                ])

            camera_name = f"cam_{i + 1:02d}"
            quaternion_wxyz = look_at_quaternion(cam_pos, self.aim_point)

            camera = Camera(
                prim_path=f"/World/{camera_name}",
                position=cam_pos,
                frequency=frequency,
                resolution=resolution,
                orientation=quaternion_wxyz,
            )
            self.cameras.append(camera)

            if self.verbose:
                distance = np.linalg.norm(self.aim_point - cam_pos)
                print(f"[CAMS] {camera_name}: pos={cam_pos}, distance_to_center={distance:.2f}")
    
    def _place_cameras_lgm(self, cfg, frequency, resolution):
        """4 LGM views at azimuth [0, 90, 180, 270] deg plus offset, fixed elevation, looking at center."""
        lgm_cfg = self.cfg.get("lgm", {})
        radius = float(lgm_cfg.get("radius", 25.0))
        elevation_deg = float(lgm_cfg.get("elevation", 0.0))
        center = list(lgm_cfg.get("center", [0.0, 0.0, 0.0]))

        y_up = self.scene_params and self.scene_params['up_axis'] == 'Y'

        if self.scene_params and self.scene_params['meters_per_unit'] != 1.0:
            scale_factor = 1.0 / self.scene_params['meters_per_unit']
            radius *= scale_factor
            center = [c * scale_factor for c in center]

        self.aim_point = np.array(center)
        elevation_rad = np.radians(elevation_deg)

        az_offset = float(lgm_cfg.get("azimuth_offset", 0.0))
        azimuths_deg = [az_offset + a for a in [0.0, 90.0, 180.0, 270.0]]

        if self.verbose:
            axis_label = "Y-up" if y_up else "Z-up"
            print(f"[CAMS] LGM placement: 4 cameras, radius={radius}, "
                  f"elevation={elevation_deg} deg, az_offset={az_offset} deg, "
                  f"center={center} ({axis_label})")

        for i, az_deg in enumerate(azimuths_deg):
            az_rad = np.radians(az_deg)

            # spherical to Cartesian
            horiz = radius * cos(elevation_rad)
            vert = radius * sin(elevation_rad)

            if y_up:
                cam_pos = np.array([
                    horiz * cos(az_rad) + center[0],
                    vert + center[1],
                    horiz * sin(az_rad) + center[2],
                ])
            else:
                cam_pos = np.array([
                    horiz * cos(az_rad) + center[0],
                    horiz * sin(az_rad) + center[1],
                    vert + center[2],
                ])

            camera_name = f"cam_{i + 1:02d}"
            quaternion_wxyz = look_at_quaternion(cam_pos, self.aim_point)

            camera = Camera(
                prim_path=f"/World/{camera_name}",
                position=cam_pos,
                frequency=frequency,
                resolution=resolution,
                orientation=quaternion_wxyz,
            )
            self.cameras.append(camera)

            if self.verbose:
                dist = np.linalg.norm(self.aim_point - cam_pos)
                print(f"[CAMS] {camera_name}: az={az_deg} deg, el={elevation_deg} deg, "
                      f"pos={cam_pos}, dist={dist:.2f}")

    def _configure_intrinsics(self, intrinsics_cfg):
        """Apply intrinsics to all cameras; focus at aim point."""
        for i, camera in enumerate(self.cameras):
            if self.aim_point is not None:
                camera_distance = np.linalg.norm(self.aim_point - np.array(camera.get_world_pose()[0]))
                configure_camera_intrinsics(camera, intrinsics_cfg, camera_distance)


def configure_camera_intrinsics(camera, intrinsics_config, camera_distance):
    """Focal length, apertures, focus, distortion from intrinsics_config."""
    if not intrinsics_config:
        return
    
    width, height = camera.get_resolution()
    pixel_size = float(intrinsics_config.get("pixel_size", 3.0))
    f_stop = float(intrinsics_config.get("f_stop", 1.8))
    focus_distance = float(intrinsics_config.get("focus_distance", camera_distance))
    camera_matrix = intrinsics_config.get("camera_matrix", None)
    distortion_model = intrinsics_config.get("distortion_model", "pinhole")
    distortion_coeffs = intrinsics_config.get("distortion_coefficients", None)
    focal_length_override = intrinsics_config.get("focal_length", None)
    horizontal_aperture_override = intrinsics_config.get("horizontal_aperture", None)
    vertical_aperture_override = intrinsics_config.get("vertical_aperture", None)

    # physical params from K (pixel_size in um), else defaults
    if camera_matrix is not None:
        ((fx, _, cx), (_, fy, cy), (_, _, _)) = camera_matrix

        horizontal_aperture = pixel_size * width * 1e-4    # um to cm
        vertical_aperture = pixel_size * height * 1e-4    # um to cm

        focal_length_x = pixel_size * fx * 1e-4           # um to cm
        focal_length_y = pixel_size * fy * 1e-4           # um to cm
        focal_length = (focal_length_x + focal_length_y) / 2.0
        
    else:
        horizontal_aperture = horizontal_aperture_override or (pixel_size * width * 1e-4)
        vertical_aperture = vertical_aperture_override or (pixel_size * height * 1e-4)
        focal_length = focal_length_override or 2.4       # cm

        fx = focal_length / (pixel_size * 1e-4)
        fy = fx
        cx = width / 2.0
        cy = height / 2.0

    camera.set_focal_length(focal_length)
    camera.set_focus_distance(focus_distance)
    camera.set_lens_aperture(f_stop)
    camera.set_horizontal_aperture(horizontal_aperture)
    camera.set_vertical_aperture(vertical_aperture)

    # skip OpenCV pinhole override when coefficients all zero: physical
    # params already give correct FOV, override can break projection
    if distortion_coeffs is not None and camera_matrix is not None:
        ((fx, _, cx), (_, fy, cy), (_, _, _)) = camera_matrix
        has_distortion = any(abs(c) > 0 for c in distortion_coeffs)

        if distortion_model == "pinhole" and has_distortion:
            camera.set_opencv_pinhole_properties(cx=cx, cy=cy, fx=fx, fy=fy, pinhole=distortion_coeffs)
        elif distortion_model == "fisheye":
            camera.set_opencv_fisheye_properties(cx=cx, cy=cy, fx=fx, fy=fy, fisheye=distortion_coeffs)
        elif distortion_model == "kannala_brandt":
            nominal_width = float(width)
            nominal_height = float(height)
            optical_centre_x = cx
            optical_centre_y = cy
            camera.set_kannala_brandt_properties(
                nominal_width=nominal_width,
                nominal_height=nominal_height,
                optical_centre_x=optical_centre_x,
                optical_centre_y=optical_centre_y,
                max_fov=None,
                distortion_model=distortion_coeffs
            )
        elif distortion_model == "rational_polynomial":
            nominal_width = float(width)
            nominal_height = float(height)
            optical_centre_x = cx
            optical_centre_y = cy
            camera.set_rational_polynomial_properties(
                nominal_width=nominal_width,
                nominal_height=nominal_height,
                optical_centre_x=optical_centre_x,
                optical_centre_y=optical_centre_y,
                max_fov=None,
                distortion_model=distortion_coeffs
            )


def initialize_cameras(cameras, scene_params=None, intrinsics_config=None, aim_point=None):
    """world.reset() then camera.initialize() for each."""
    if not cameras:
        print("[CAMS] No cameras to initialize")
        return
    
    stage_units = scene_params['stage_units'] if scene_params else 1.0
    try:
        world = World.instance()
        if world is None:
            world = World(stage_units_in_meters=stage_units)
    except:
        world = World(stage_units_in_meters=stage_units)
    
    world.reset()
    
    for camera in cameras:
        camera.initialize()
        pos, rot = camera.get_world_pose()
        print(f"[CAMS] {camera.name} initialized at position: {pos}")
