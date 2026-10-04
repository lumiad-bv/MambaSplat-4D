"""Gaussian lifting (from VN-3DGS/src/models/gaussian_lifting.py)."""
import torch
import torch.nn.functional as F
from typing import Optional


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """(..., 4) [w, x, y, z] -> (..., 3, 3)."""
    q = F.normalize(q, p=2, dim=-1)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    R = torch.stack([
        torch.stack([1 - 2*(y**2 + z**2), 2*(x*y - w*z),     2*(x*z + w*y)],     dim=-1),
        torch.stack([2*(x*y + w*z),       1 - 2*(x**2 + z**2), 2*(y*z - w*x)],     dim=-1),
        torch.stack([2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x**2 + y**2)], dim=-1),
    ], dim=-2)

    return R


def gaussian_lifting(
    position: torch.Tensor,
    quaternion: Optional[torch.Tensor],
    scale: Optional[torch.Tensor],
    use_centroid: bool = True,
    use_rotation_axes: bool = True,
    use_scale_weighting: bool = True,
) -> torch.Tensor:
    """Gaussians -> equivariant vectors (B, N, K, 3): centroid + optional
    scale-weighted principal axes. quaternion=None skips axes."""
    vectors = []

    if use_centroid:
        vectors.append(position)

    if use_rotation_axes and quaternion is not None:
        R_q = quaternion_to_rotation_matrix(quaternion)
        e1 = R_q[..., :, 0]
        e2 = R_q[..., :, 1]
        e3 = R_q[..., :, 2]

        if use_scale_weighting and scale is not None:
            e1 = scale[..., 0:1] * e1
            e2 = scale[..., 1:2] * e2
            e3 = scale[..., 2:3] * e3

        vectors.append(e1)
        vectors.append(e2)
        vectors.append(e3)

    if len(vectors) == 0:
        raise ValueError(
            "At least one of use_centroid or use_rotation_axes must be True"
        )

    return torch.stack(vectors, dim=2)


def count_lifting_vectors(
    use_centroid: bool = True,
    use_rotation_axes: bool = True,
) -> int:
    """K for gaussian_lifting."""
    count = 0
    if use_centroid:
        count += 1
    if use_rotation_axes:
        count += 3
    return count
