"""SO(3) sampling and quaternion ops (from VN-3DGS/src/utils/rotations.py).
Haar-uniform sampling via vendored PyTorch3D (utils/pytorch3d_transforms)."""
from typing import Optional

import numpy as np
import torch
from torch import sin, cos


def rot_z(gamma: torch.Tensor) -> torch.Tensor:
    """Rz."""
    return torch.tensor([
        [cos(gamma), -sin(gamma), 0],
        [sin(gamma), cos(gamma), 0],
        [0, 0, 1]
    ], dtype=gamma.dtype)


def rot_y(beta: torch.Tensor) -> torch.Tensor:
    """Ry."""
    return torch.tensor([
        [cos(beta), 0, sin(beta)],
        [0, 1, 0],
        [-sin(beta), 0, cos(beta)]
    ], dtype=beta.dtype)


def rot(alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """ZYZ Euler -> R."""
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)


def random_rotation_matrix(device: torch.device = None) -> torch.Tensor:
    """Haar-uniform SO(3) matrix (torch, vendored PyTorch3D)."""
    # lazy: avoid pytorch3d import cost
    from mambasplat4d.utils.pytorch3d_transforms import random_rotation
    return random_rotation(device=device)


def random_rotation_matrix_np(
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Haar-uniform SO(3) matrix (numpy mirror of pytorch3d.random_rotations; Shoemake 1992):
    4D Gaussian -> unit quaternion -> R. Torch-free for DataLoader workers."""
    gen = rng if rng is not None else np.random
    q = gen.standard_normal(4).astype(np.float32)
    q /= max(float(np.linalg.norm(q)), 1e-8)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """wxyz quaternion -> R."""
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    R = torch.stack([
        torch.stack([1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w], dim=-1),
        torch.stack([2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w], dim=-1),
        torch.stack([2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2], dim=-1),
    ], dim=-2)

    return R
