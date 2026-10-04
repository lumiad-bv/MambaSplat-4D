"""Deterministic seeding: Python, NumPy, torch CPU + all GPUs, cudnn, deterministic algorithms."""
import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True):
    """deterministic=True: cudnn.benchmark off, use_deterministic_algorithms on (slower, bit-exact)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # required by deterministic CUBLAS
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
