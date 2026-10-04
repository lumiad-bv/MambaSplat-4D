"""Asset-token helpers shared by reconstruction and preprocessing.

Sequence folder (Isaac Sim renderer):
    blank_lgm_1.4r_20el_45az_4cams_lgm_pinhole_5.0x_20_Owl_merged
Asset token = substring after last `_<speed>x_`:
    20_Owl_merged

Tokens come from .usdc filenames in `config_batch_lgm*.yaml`, with spaces, '+', '.'
replaced by '_'. Hyphens kept. Leading digits are part of the name; never strip.

Token decides split membership: this regex is the single source of truth for the
identity-disjoint split.
"""
import re

# Speed marker before asset token: "_5.0x_", "_1.5x_", "_10x_".
_ASSET_TOKEN_RE = re.compile(r'_\d+(?:\.\d+)?x_(.+)$')

# Sim-3 line formation: token in middle, speed marker at end, e.g.
#   rivermark_lform_1956_L-1049G_Super_Constellation_Full_Interior_custom_1440p_30deg_pinhole_3.0x
# Token sits between "rivermark_lform_" and "_custom_..._pinhole_<speed>x".
_LFORM_TOKEN_RE = re.compile(
    r'^rivermark_lform_(.+)_custom_.*_pinhole_\d+(?:\.\d+)?x$'
)


def extract_asset_token(seq_name: str) -> str:
    """Asset token from sequence folder name.

    >>> extract_asset_token('blank_lgm_1.4r_20el_45az_4cams_lgm_pinhole_5.0x_20_Owl_merged')
    '20_Owl_merged'
    >>> extract_asset_token('blank_lgm_10.0r_0el_0az_4cams_lgm_pinhole_5.0x_animated-stylized-tukan-3d-animal-model')
    'animated-stylized-tukan-3d-animal-model'
    >>> extract_asset_token('rivermark_lform_1956_L-1049G_Super_Constellation_Full_Interior_custom_1440p_30deg_pinhole_3.0x')
    '1956_L-1049G_Super_Constellation_Full_Interior'

    Raises ValueError if neither `_<speed>x_<asset>` nor
    `rivermark_lform_<asset>_..._pinhole_<speed>x` matches.
    """
    m = _ASSET_TOKEN_RE.search(seq_name)
    if m:
        return m.group(1)
    m = _LFORM_TOKEN_RE.match(seq_name)
    if m:
        return m.group(1)
    raise ValueError(
        f"Cannot extract asset token from sequence name: {seq_name!r}. "
        "Expected a '_<speed>x_<asset>' suffix or a 'rivermark_lform_<asset>_..._<speed>x' name."
    )


# Reconstructed frame basenames: 'frame_0000.ply' / 'frame_0000_s0.pt' (optional sweep
# suffix), optional '<seq>_' prefix.
_FRAME_RE = re.compile(r'frame_(\d+)(?:_s(\d+))?\.(ply|pt)$')


def parse_frame_filename(fname: str) -> tuple[int, int | None, str]:
    """Parse `frame_XXXX[_sY].{ply,pt}` into (frame_idx, sweep_idx|None, ext).

    >>> parse_frame_filename('frame_0003.ply')
    (3, None, 'ply')
    >>> parse_frame_filename('frame_0012_s1.pt')
    (12, 1, 'pt')
    """
    m = _FRAME_RE.search(fname)
    if not m:
        raise ValueError(f"Not a recognised frame file: {fname!r}")
    return int(m.group(1)), (int(m.group(2)) if m.group(2) else None), m.group(3)
