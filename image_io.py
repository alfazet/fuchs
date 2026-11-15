from __future__ import annotations

import numpy as np
from PIL import Image

GRID_SIZE = 256
BLOCK_SIZE = 4
BITS_PER_SIDE = GRID_SIZE // BLOCK_SIZE

_LUMA_COEFFS = np.array([0.299, 0.587, 0.114], dtype=np.float64)


def _load_luma(path) -> np.ndarray:
    """
    Load an image and return the matrix of the luma of its pixels.
    Transparent pixels are treated as white.
    """
    with Image.open(path) as img:
        rgba = img.convert("RGBA")
    data = np.asarray(rgba, dtype=np.float64) / 255.0
    rgb, alpha = data[..., :3], data[..., 3:4]
    rgb = rgb * alpha + (1.0 - alpha)

    return rgb @ _LUMA_COEFFS


def read_bits(path, threshold: float = 0.5) -> np.ndarray:
    """
    Read a PNG file and decode it into a (BITS_PER_SIDE, BITS_PER_SIDE) array of bits.
    The top-left corner of the grid is the intersection of the topmost row and
    the leftmost column containing a pixel with luma <= `threshold`. Decoding starts
    from there and anything beyond GRID_SIZE x GRID_SIZE is ignored.

    Each BLOCK_SIZE x BLOCK_SIZE block of pixels is decoded as a 0 if its
    mean luma is above `threshold`, and 1 otherwise.
    """
    luma = _load_luma(path)
    dark = luma <= threshold
    if not dark.any():
        raise ValueError("invalid data (no dark pixels found)")

    top = int(np.flatnonzero(dark.any(axis=1))[0])
    left = int(np.flatnonzero(dark.any(axis=0))[0])

    data_region = luma[top:, left:]
    height, width = data_region.shape
    if height < GRID_SIZE or width < GRID_SIZE:
        raise ValueError(
            f"decoding area {width} x {height} too small (required minimum is {GRID_SIZE} x {GRID_SIZE})"
        )
    data_region = data_region[:GRID_SIZE, :GRID_SIZE]
    blocks = data_region.reshape(BITS_PER_SIDE, BLOCK_SIZE, BITS_PER_SIDE, BLOCK_SIZE)
    mean = blocks.mean(axis=(1, 3))

    return (mean <= threshold).astype(np.uint8)


def write_bits(bits: np.ndarray, path) -> None:
    """
    Save a (BITS_PER_SIDE, BITS_PER_SIDE) array of bits as a GRID_SIZE x GRID_SIZE PNG file.
    Each bit becomes a BLOCK_SIZE x BLOCK_SIZE block of white (0) or dark (1) pixels.
    """
    arr = np.asarray(bits)
    expected = (BITS_PER_SIDE, BITS_PER_SIDE)
    if arr.shape != expected:
        raise ValueError(f"expected bit array of shape {expected} (got {arr.shape})")
    if not np.isin(arr, (0, 1)).all():
        raise ValueError("bit array must contain only 0s and 1s")

    pixels = np.kron(
        arr.astype(np.uint8), np.ones((BLOCK_SIZE, BLOCK_SIZE), dtype=np.uint8)
    )
    image = Image.fromarray((1 - pixels) * 255, mode="L")
    image.save(path, format="PNG")
