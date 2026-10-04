"""Gaussian feature modes (Table 2 ablation). C center, O opacity, S scale,
R quaternion, SH spherical harmonics DC, X = all.

Mode     flat dim  invariant dim
C           3        0
C_O         4        1 (opacity)
C_SH        6        3 (sh_dc)
C_S_R      10        0 (scale weights axes)
C_O_S_R    11        1 (opacity)
X          14        4 (opacity + sh_dc)
"""

FEATURE_MODES = {
    'C':       ('position',),
    'C_O':     ('position', 'opacity'),
    'C_SH':    ('position', 'sh_dc'),
    'C_S_R':   ('position', 'scale', 'quaternion'),
    'C_O_S_R': ('position', 'opacity', 'scale', 'quaternion'),
    'X':       ('position', 'opacity', 'scale', 'quaternion', 'sh_dc'),
}

_KEY_DIMS = {
    'position': 3, 'opacity': 1, 'scale': 3, 'quaternion': 4, 'sh_dc': 3,
}

FEATURE_DIMS = {
    mode: sum(_KEY_DIMS[k] for k in keys)
    for mode, keys in FEATURE_MODES.items()
}

# concatenation order
ALL_GAUSSIAN_KEYS = ('position', 'quaternion', 'scale', 'opacity', 'sh_dc')

FEATURE_MODE_ORDER = ['C', 'C_O', 'C_SH', 'C_S_R', 'C_O_S_R', 'X']


def has_rotation_axes(feature_mode: str) -> bool:
    """Quaternion present: rotation axes lifted."""
    return 'quaternion' in FEATURE_MODES[feature_mode]


def has_scale_weighting(feature_mode: str) -> bool:
    """Scale and quaternion present: scale weights axes."""
    keys = FEATURE_MODES[feature_mode]
    return 'scale' in keys and 'quaternion' in keys


def get_invariant_dim(feature_mode: str) -> int:
    """Invariant scalar dim: opacity (1) + sh_dc (3) + scale (3) if not axis-weighted."""
    keys = set(FEATURE_MODES[feature_mode])
    dim = 0
    if 'opacity' in keys:
        dim += 1
    if 'sh_dc' in keys:
        dim += 3
    if 'scale' in keys and not has_scale_weighting(feature_mode):
        dim += 3
    return dim


def get_feature_dim(feature_mode: str) -> int:
    """Flat feature dim."""
    return FEATURE_DIMS[feature_mode]
