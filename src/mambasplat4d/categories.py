"""Single source of truth for AeroSplat-4D class ids (reconstruction, preprocessing, training)."""

CATEGORIES: dict[str, int] = {
    "bird": 0,
    "drone": 1,
    "airplane": 2,
    "helicopter": 3,
}

CATEGORY_NAMES: dict[int, str] = {v: k for k, v in CATEGORIES.items()}
