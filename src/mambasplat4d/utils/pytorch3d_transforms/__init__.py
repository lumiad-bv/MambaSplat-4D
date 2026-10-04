"""Vendored subset of Meta's PyTorch3D for Haar-uniform SO(3) sampling.

Only `rotation_conversions.py` is vendored (not `so3.py`, not `math.py`) because
only the Haar-sampling helpers and a few quaternion conversions are needed.
BSD-3 license preserved in ./LICENSE and in the file header.

Upstream: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py

Replaces the previous inline axis-angle sampler (angle ~ Uniform[0, 2π]), which
was not Haar-uniform on SO(3): the correct marginal angle density is
(1 - cos θ)/π on [0, π] (Miles 1965). Quaternion-based sampling via 4D Gaussian
normalization (Shoemake 1992) sidesteps the non-closed-form CDF inversion.
"""
from .rotation_conversions import (
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    quaternion_apply,
    quaternion_invert,
    quaternion_multiply,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
    random_quaternions,
    random_rotation,
    random_rotations,
    standardize_quaternion,
)

__all__ = [
    "axis_angle_to_matrix",
    "axis_angle_to_quaternion",
    "matrix_to_axis_angle",
    "matrix_to_quaternion",
    "quaternion_apply",
    "quaternion_invert",
    "quaternion_multiply",
    "quaternion_to_axis_angle",
    "quaternion_to_matrix",
    "random_quaternions",
    "random_rotation",
    "random_rotations",
    "standardize_quaternion",
]
