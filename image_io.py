import cv2
import numpy as np
from PIL import Image

BLOCK_SIZE = 4   # only used by write_bits
B = 16           # bit size (px) after straightening the photo
R = 0.25         # thickness of one frame ring as a fraction of a bit (same as in write_bits)


def _line(P):
    """Line fitted to points: (point, direction)."""
    vx, vy, x, y = cv2.fitLine(P, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return np.array([x, y]), np.array([vx, vy])


def _cross(l1, l2):
    """Intersection of two lines."""
    (p1, d1), (p2, d2) = l1, l2
    t = np.linalg.solve(np.array([d1, -d2]).T, p2 - p1)[0]
    return p1 + t * d1


def _find_quad(gray):
    """Find the 4 corners of the outer black frame: tl, tr, br, bl."""
    bw = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255,
                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    k = max(3, gray.shape[1] // 200)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)

    for eps in (0.01, 0.02, 0.04, 0.08):
        approx = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            break
    else:  # no quadrilateral found -> plain rotated rectangle
        pts = cv2.boxPoints(cv2.minAreaRect(hull))

    s, d = pts.sum(axis=1), pts[:, 1] - pts[:, 0]
    quad = np.float32([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]])

    # Corners are rounded after blurring, so refine them: fit a line to each side
    # (ignoring the ends near the corners) and take intersections of neighbouring lines.
    P = cnt.reshape(-1, 2).astype(np.float32)
    lines = []
    for a, b in zip(quad, np.roll(quad, -1, axis=0)):
        L = np.linalg.norm(b - a)
        d = (b - a) / L
        rel = (P - a) @ d / L
        dist = np.abs((P - a) @ np.array([-d[1], d[0]]))
        m = (rel > 0.15) & (rel < 0.85) & (dist < 0.03 * L)
        if m.sum() < 10:
            return quad
        lines.append(_line(P[m]))
    return np.float32([_cross(lines[i - 1], lines[i]) for i in range(4)])


def _warp(gray, quad, w, h):
    """Straighten the quadrilateral into a w x h rectangle (fixes tilt and perspective)."""
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    return cv2.warpPerspective(gray, cv2.getPerspectiveTransform(quad, dst), (w, h))


def _count_bits(g):
    """
    How many bits does the top side have? Try every possible n and check for which one
    the black/white frame matches the pattern 'white, black, white, ...' (as in write_bits).
    A wrong n shifts the pattern, so its score is close to zero.
    The top and bottom frames are checked together so that a random match with the data doesn't win.
    """
    L = g.shape[1]
    mid = (np.percentile(g, 5) + np.percentile(g, 95)) / 2   # midpoint between black and white
    best_score, best_n = -1e9, 0
    for n in range(2, 400):
        b = L / (n + 4 * R)                      # bit size assuming n bits
        if b < 3:
            break
        y0 = int(1.2 * R * b)                    # middle of the ring with alternating colors
        y1 = max(y0 + 1, int(1.8 * R * b))
        idx = (2 * R * b + (np.arange(n) + 0.5) * b).astype(int)   # centers of the frame segments
        sign = 1 - 2 * (np.arange(n) % 2)        # +1 white, -1 black
        score = 0
        for side in (g, g[::-1]):                # top and bottom
            prof = side[y0:y1].mean(axis=0)
            score += (sign * np.sign(prof[idx] - mid)).mean()   # only sign agreement, no amplitude
        score *= np.sqrt(n)                      # so that a small n doesn't win by chance
        if score > best_score:
            best_score, best_n = score, n
    return best_n


def read_bits(path) -> np.ndarray:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    quad = _find_quad(gray)
    tl, tr, br, bl = quad

    # 1. rough straightening (size from side lengths) and counting the bits
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    rough = _warp(gray, quad, w, h)
    W, H = _count_bits(rough), _count_bits(rough.T)

    # 2. exact straightening: every bit is exactly B x B pixels
    img = _warp(gray, quad, int((W + 4 * R) * B), int((H + 4 * R) * B))

    # 3. reading: mean of the central half of each bit
    o, q = int(2 * R * B), B // 4
    cells = img[o:o + H * B, o:o + W * B].reshape(H, B, W, B)[:, q:-q, :, q:-q].mean(axis=(1, 3))
    thr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    return (cells < thr).astype(np.uint8)


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
