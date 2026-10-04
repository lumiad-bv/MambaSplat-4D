"""FPS from index 0, squared Euclidean. Matches pointnet2_ops.furthest_point_sample,
incl. never selecting points with squared norm <= 1e-3."""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """xyz (N, 3) -> (npoint,) int64 indices."""
    N = xyz.shape[0]
    if npoint >= N:
        return torch.arange(N, dtype=torch.long, device=xyz.device)
    idx = torch.empty(npoint, dtype=torch.long, device=xyz.device)
    dist = torch.full((N,), float("inf"), dtype=torch.float32, device=xyz.device)
    xyz = xyz.float()
    dist[(xyz ** 2).sum(dim=1) <= 1e-3] = float("-inf")
    cur = 0
    for i in range(npoint):
        idx[i] = cur
        d = ((xyz - xyz[cur]) ** 2).sum(dim=1)
        dist = torch.minimum(dist, d)
        cur = int(torch.argmax(dist))
    return idx


def farthest_point_sample_np(xyz: np.ndarray, npoint: int) -> np.ndarray:
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    N = xyz.shape[0]
    if N == 0:
        return np.zeros((0,), dtype=np.int64)
    if npoint >= N:
        return np.arange(N, dtype=np.int64)
    pts = torch.from_numpy(xyz)
    if torch.cuda.is_available():
        pts = pts.cuda(non_blocking=True)
    return farthest_point_sample(pts, npoint).cpu().numpy().astype(np.int64)
