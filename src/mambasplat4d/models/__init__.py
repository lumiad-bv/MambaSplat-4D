"""Models package; import registers encoders."""
from . import spatial  # noqa: F401
from . import temporal  # noqa: F401
from .registry import build_spatial_encoder, build_temporal_encoder
