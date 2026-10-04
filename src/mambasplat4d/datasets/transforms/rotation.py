"""Rotation, sampling and augmentation transforms for Gaussian data."""
import numpy as np
import torch
from typing import Dict, Tuple, Sequence

from mambasplat4d.utils.fps import farthest_point_sample_np


def _build_fps_feature(data: Dict[str, np.ndarray], attribute: Sequence[str]) -> np.ndarray:
    """Feature tensor for FPS distances."""
    parts = []
    if "xyz" in attribute and "position" in data:
        parts.append(data["position"])
    if "opacity" in attribute and "opacity" in data:
        parts.append(data["opacity"])
    if "scale" in attribute and "scale" in data:
        parts.append(data["scale"])
    if "rotation" in attribute and "quaternion" in data:
        parts.append(data["quaternion"])
    if "sh" in attribute and "sh_dc" in data:
        parts.append(data["sh_dc"])

    if not parts:
        parts = [data["position"]]

    return np.ascontiguousarray(np.concatenate(parts, axis=-1), dtype=np.float32)


def random_rotation_matrix() -> np.ndarray:
    """Haar SO(3) matrix (numpy). Compat wrapper; new code calls `random_rotation_matrix_np`."""
    from mambasplat4d.utils.rotation_utils import random_rotation_matrix_np
    return random_rotation_matrix_np()


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion [w, x, y, z]."""
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float32)
    return q / np.linalg.norm(q)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion product, [w, x, y, z]."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    q = np.stack([w, x, y, z], axis=-1)
    return (q / np.linalg.norm(q, axis=-1, keepdims=True)).astype(np.float32)


def rotate_quaternion(q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotate quaternions q by R."""
    q_R = rotation_matrix_to_quaternion(R)
    return quaternion_multiply(q_R[np.newaxis, :], q)


class RandomSO3Rotation:
    """Random SO(3) rotation."""

    def __init__(self, p: float = 1.0):
        self.p = p

    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if np.random.random() > self.p:
            return data

        R = random_rotation_matrix()
        out = dict(data)
        out['position'] = data['position'] @ R.T
        if 'quaternion' in data:
            out['quaternion'] = rotate_quaternion(data['quaternion'], R)
        return out


class RandomZRotation:
    """Random z-axis rotation."""

    def __init__(self, p: float = 1.0):
        self.p = p

    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if np.random.random() > self.p:
            return data

        angle = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        out = dict(data)
        out['position'] = data['position'] @ R.T
        if 'quaternion' in data:
            out['quaternion'] = rotate_quaternion(data['quaternion'], R)
        return out


class RandomJitter:
    """Gaussian position jitter."""

    def __init__(self, std: float = 0.01, p: float = 0.5):
        self.std = std
        self.p = p

    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if np.random.random() > self.p:
            return data
        data = dict(data)
        data['position'] = data['position'] + np.random.randn(*data['position'].shape).astype(np.float32) * self.std
        return data


class RandomScale:
    """Random scale, uniform or per-axis.

    per_axis: diag(sx, sy, sz) on 'position' and 'scale'. Anisotropic, does not
    commute with SO(3); keep off under SO(3) augmentation.
    """

    def __init__(self, scale_range: tuple = (0.8, 1.2), p: float = 0.5,
                 per_axis: bool = False):
        self.scale_range = scale_range
        self.p = p
        self.per_axis = per_axis

    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if np.random.random() > self.p:
            return data
        data = dict(data)
        if self.per_axis:
            scale = np.random.uniform(*self.scale_range, size=3).astype(np.float32)
            data['position'] = data['position'] * scale
            if 'scale' in data:
                data['scale'] = data['scale'] * scale
        else:
            scale = np.random.uniform(*self.scale_range)
            data['position'] = data['position'] * scale
            if 'scale' in data:
                data['scale'] = data['scale'] * scale
        return data


class RandomTranslate:
    """Random per-axis translation."""

    def __init__(self, translate_range: Tuple[float, float] = (-0.2, 0.2), p: float = 0.5):
        self.translate_range = translate_range
        self.p = p

    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if np.random.random() > self.p:
            return data
        data = dict(data)
        t = np.random.uniform(self.translate_range[0], self.translate_range[1], size=3).astype(np.float32)
        data['position'] = data['position'] + t
        return data


class RandomDropPoints:
    """Drop a fraction of points."""

    def __init__(self, drop_ratio: float = 0.1, p: float = 0.5):
        self.drop_ratio = drop_ratio
        self.p = p

    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if np.random.random() > self.p:
            return data
        N = data['position'].shape[0]
        keep_n = max(1, int(N * (1 - self.drop_ratio)))
        indices = np.random.choice(N, keep_n, replace=False)
        return {key: arr[indices] for key, arr in data.items()}


class RandomInputDropout:
    """Shape-preserving point dropout (SimCLR-style)."""

    def __init__(self, max_dropout_ratio: float = 0.2, p: float = 0.5):
        if max_dropout_ratio < 0 or max_dropout_ratio >= 1:
            raise ValueError("max_dropout_ratio must be in [0, 1)")
        self.max_dropout_ratio = max_dropout_ratio
        self.p = p

    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if np.random.random() > self.p:
            return data

        N = data['position'].shape[0]
        if N <= 1:
            return data

        dropout_ratio = np.random.random() * self.max_dropout_ratio
        drop_idx = np.where(np.random.random(N) <= dropout_ratio)[0]
        if len(drop_idx) == 0:
            return data

        out = dict(data)
        for key, arr in out.items():
            arr = arr.copy()
            arr[drop_idx] = arr[0]
            out[key] = arr
        return out


class RandomSample:
    """Random subsample without replacement."""

    def __init__(self, num_points: int = 1024):
        self.num_points = num_points

    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        N = data['position'].shape[0]
        if N <= self.num_points:
            return data
        indices = np.random.choice(N, self.num_points, replace=False)
        return {key: arr[indices] for key, arr in data.items()}


class RandomSampleReplace:
    """Random sample with replacement (G-MAE ModelNet loader)."""

    def __init__(self, num_points: int = 8192):
        self.num_points = num_points

    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        N = data["position"].shape[0]
        if N == 0:
            return data
        indices = np.random.choice(N, self.num_points, replace=True)
        return {key: arr[indices] for key, arr in data.items()}


class FPSByAttribute:
    """FPS on selected attributes (xyz/opacity/scale/rotation/sh)."""

    def __init__(self, num_points: int = 1024, attribute: Sequence[str] = ("xyz",)):
        self.num_points = num_points
        self.attribute = tuple(attribute)

    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        N = data["position"].shape[0]
        if N == 0:
            return data
        if N < self.num_points:
            indices = np.random.choice(N, self.num_points, replace=True)
            return {key: arr[indices] for key, arr in data.items()}
        if N == self.num_points:
            return data

        fps_feature = _build_fps_feature(data, self.attribute)
        indices = farthest_point_sample_np(fps_feature, self.num_points)
        return {key: arr[indices] for key, arr in data.items()}


class FPSRandomSubsample:
    """G-MAE sampling: FPS to point_all, then uniform subset to num_points."""

    def __init__(
        self,
        num_points: int = 1024,
        point_all: int = 1200,
        attribute: Sequence[str] = ("xyz",),
    ):
        self.num_points = num_points
        self.point_all = point_all
        self.attribute = tuple(attribute)

    @staticmethod
    def _default_point_all(num_points: int) -> int:
        if num_points == 1024:
            return 1200
        if num_points == 2048:
            return 2400
        if num_points == 4096:
            return 4800
        if num_points == 8192:
            return 8192
        return num_points

    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        N = data["position"].shape[0]
        if N == 0:
            return data

        point_all = self.point_all if self.point_all > 0 else self._default_point_all(self.num_points)
        point_all = min(point_all, N)
        if point_all <= 0:
            return data

        if N > point_all:
            fps_feature = _build_fps_feature(data, self.attribute)
            fps_idx = farthest_point_sample_np(fps_feature, point_all)
        else:
            fps_idx = np.arange(N, dtype=np.int64)

        replace = point_all < self.num_points
        choice = np.random.choice(point_all, self.num_points, replace=replace)
        final_idx = fps_idx[choice]
        return {key: arr[final_idx] for key, arr in data.items()}


class FPSSample:
    """Greedy FPS on xyz."""

    def __init__(self, num_points: int = 1024):
        self.num_points = num_points

    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        N = data['position'].shape[0]
        if N <= self.num_points:
            return data
        positions = np.ascontiguousarray(data['position'], dtype=np.float32)
        indices = farthest_point_sample_np(positions, self.num_points)
        return {key: arr[indices] for key, arr in data.items()}


class RandomCrop:
    """Local crop, resampled to original N."""

    def __init__(self, keep_ratio: Tuple[float, float] = (0.7, 0.9), p: float = 0.3):
        self.keep_ratio = keep_ratio
        self.p = p

    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if np.random.random() > self.p:
            return data

        N = data['position'].shape[0]
        if N <= 4:
            return data

        ratio = float(np.random.uniform(self.keep_ratio[0], self.keep_ratio[1]))
        keep_n = int(max(1, min(N, round(N * ratio))))

        anchor_idx = np.random.randint(0, N)
        anchor = data['position'][anchor_idx:anchor_idx + 1]
        d2 = np.sum((data['position'] - anchor) ** 2, axis=-1)

        keep_idx = np.argpartition(d2, keep_n - 1)[:keep_n]
        if keep_n < N:
            extra = np.random.choice(keep_idx, size=N - keep_n, replace=True)
            final_idx = np.concatenate([keep_idx, extra], axis=0)
        else:
            final_idx = keep_idx

        np.random.shuffle(final_idx)
        return {key: arr[final_idx] for key, arr in data.items()}


class CenterNormalize:
    """Center at origin, scale to unit sphere."""

    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out = dict(data)
        pos = out['position'].astype(np.float32, copy=True)
        center = pos.mean(axis=0, keepdims=True)
        pos = pos - center
        radius = np.linalg.norm(pos, axis=-1).max()
        if radius < 1e-8:
            radius = 1.0
        inv = 1.0 / float(radius)
        pos = pos * inv
        out['position'] = pos
        if 'scale' in out:
            out['scale'] = out['scale'] * inv
        return out


class Compose:
    """Chain transforms."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for t in self.transforms:
            data = t(data)
        return data


def get_train_sampling_transforms(
    num_points: int = 1024,
    jitter_std: float = 0.01,
    jitter_p: float = 0.5,
    scale_range: tuple = (0.8, 1.25),
    scale_p: float = 0.5,
    scale_per_axis: bool = False,
    drop_ratio: float = 0.1,
    drop_p: float = 0.5,
) -> Compose:
    """Train: random sample + augmentation."""
    return Compose([
        RandomSample(num_points=num_points),
        RandomDropPoints(drop_ratio=drop_ratio, p=drop_p),
        RandomJitter(std=jitter_std, p=jitter_p),
        RandomScale(scale_range=scale_range, p=scale_p, per_axis=scale_per_axis),
    ])


def get_eval_sampling_transforms(num_points: int = 1024) -> Compose:
    """Eval: FPS only."""
    return Compose([
        FPSSample(num_points=num_points),
    ])
