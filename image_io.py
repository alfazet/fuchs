from __future__ import annotations

import cv2
import math
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


def _save(pixels, path):
    img = Image.fromarray(pixels, mode="L")
    img.save(path, format="PNG")

def _get_longest_bars(bars: array(tuple[int, int, int, int, int])) -> array(tuple[int, int, int, int, int]):
    widest = second_widest = (0,0,0,0,0)
    dist_threshold = 100
    for b in bars:
        if b[-1] > second_widest[-1]:
            if b[-1] > widest[-1] and abs(b[1] - widest[1]) > dist_threshold:
                second_widest = widest
                widest = b
            elif abs(b[1] - widest[1]) > dist_threshold:
                second_widest = b
    return [widest, second_widest]


def _find_bars(img: np.ndarray, h: int, w: int) -> array(tuple[int, int, int, int, int]):
    step = 11
    y_step = step

    bars_up = _find_bars_up(step, y_step, img, h, w)
    bars_down = _find_bars_down(step, y_step, img, h, w)

    # Get longest found lines
    # (We assume the two longest lines are those of the rectangle frame)
    longest_up = _get_longest_bars(bars_up)
    longest_down = _get_longest_bars(bars_down)

    if longest_up[0][-1] > longest_down[0][-1]:
        return longest_up
    elif longest_up[0][-1] < longest_down[0][-1]:
        return longest_down
    else:
        if longest_up[1][-1] > longest_down[1][-1]:
            return longest_up
        else:
            return longest_down


def _find_bars_up(step: int, y_step: int, img: np.ndarray, h: int, w: int) \
    -> array(tuple[int, int, int, int, int]):
    bars = []

    def check_for_y_up(img, x, y, y_step):
        for yc in range(y_step):
            if 0 <= y - yc and img[y - yc, x] < 1:
                return (True, y - yc)
        return (False, -1)

    y = h - 1
    while y >= 0:
        new_bars = _find_bars_loop(check_for_y_up, y, step, y_step, img, h, w)
        bars = bars + new_bars
        y -= step

    return bars

def _find_bars_down(step: int, y_step: int, img: np.ndarray, h: int, w: int) \
    -> array(tuple[int, int, int, int, int]):
    bars = []

    def check_for_y_down(img, x, y, y_step):
        for yc in range(y_step):
            if y + yc < h and img[y + yc, x] < 1:
                return (True, y + yc)
        return (False, -1)

    y = 0
    while y < h:
        new_bars = _find_bars_loop(check_for_y_down, y, step, y_step, img, h, w)
        bars = bars + new_bars
        y += step

    return bars


def _find_bars_loop(check_for_y: Callable[[np.ndarray, int, int, int], tuple[bool, int]], \
    y: int, step: int, y_step: int, img: np.ndarray, h: int, w: int) \
    -> array(tuple[int, int, int, int, int]):
    bars = []
    x = 0
    while x < w:
        bar_width = 0
        bars.append((x,y,0,0,0))
        while True:
            ok, next_y = check_for_y(img, x, y, y_step)
            if not ok:
                break
            y = next_y
            bar_width += step
            x += step
            if x >= w or y >= h:
                break;
        if bar_width > w//3:
            xs, ys, _,_,_ = bars[-1]
            bars[-1] = (xs,ys,x,y,bar_width)
        else:
            bars.pop()
        x += step

    return bars


def read_bits(path) -> np.ndarray:
    img = cv2.imread(path, 0)
    h, w = img.shape
    block = w // 4 + 1 - (w //4) % 2
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 2)

    # clean up image
    kernel = np.ones((21,21), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # Find horizontal or slightly tilted lines
    bars = _find_bars(img, h, w)
    widest = bars[0]
    second_widest = bars[1]

    # Rotate image
    angle = math.degrees(math.atan2(widest[3] - widest[1], widest[2] - widest[0]))
    xs = widest[0] if widest[0] < second_widest[0] else second_widest[0]
    ys = widest[1] if widest[1] < second_widest[1] else second_widest[1]
    xe = widest[2] if widest[2] > second_widest[2] else second_widest[2]
    ye = widest[3] if widest[3] > second_widest[3] else second_widest[3]
    xc = (xs + xe) // 2
    yc = (ys + ye) // 2
    rot_matrix = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    img = cv2.warpAffine(img, rot_matrix, (w, h))
    # Rotate corner points
    pts = np.array([[[xs,ys]], [[xe,ye]]], dtype=np.float32)
    rot_pts = cv2.transform(pts, rot_matrix)
    xs, ys = rot_pts[0][0]
    xe, ye = rot_pts[1][0]
    xs, ys, xe, ye = int(xs), int(ys), int(xe), int(ye)

    # Crop image
    img = img[ys:ye, xs:xe]
    _save(img, 'output.png')


def write_bits(bits, path) -> None:
    """
    Save a 2D array of bits (0 = white, 1 = dark) as a PNG.
    Each bit is a BLOCK_SIZE x BLOCK_SIZE block, surrounded by an alternating
    black/white frame and a solid black outer frame.
    """
    arr = np.asarray(bits)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D array (got {arr.ndim}D)")
    if not np.isin(arr, (0, 1)).all():
        raise ValueError("bit array must contain only 0s and 1s")

    t = max(1, BLOCK_SIZE // 4)

    pixels = np.kron(arr.astype(np.uint8), np.ones((BLOCK_SIZE, BLOCK_SIZE), dtype=np.uint8))
    h, w = pixels.shape

    cols = (np.arange(w) // BLOCK_SIZE) % 2
    rows = (np.arange(h) // BLOCK_SIZE) % 2

    img = np.zeros((h + 2*t, w + 2*t), dtype=np.uint8)
    img[:t, t:-t] = img[-t:, t:-t] = cols
    img[t:-t, :t] = img[t:-t, -t:] = rows[:, None]
    img[:t, :t] = img[:t, -t:] = img[-t:, :t] = img[-t:, -t:] = 1
    img[t:-t, t:-t] = pixels

    img = np.pad(img, t, constant_values=1)
    Image.fromarray((1 - img) * 255).save(path)
