"""Dust & scratch detection + 16-bit inpainting for FreeCCR.

Pure ``numpy`` + OpenCV (no Qt, no scipy/skimage) so it stays Nuitka-friendly
and unit-testable headless. Operates on the CONVERTED POSITIVE (uint16 RGB),
where film dust shows as small bright specks and thin bright/dark scratches.

Heal strokes are resolution-independent vectors stored on ``CCRImage`` and
replayed through ``CCRImage.apply_adjustments`` at preview, zoom, and export
resolutions. A stroke dict::

    {"points":  [[nx, ny], ...],   # normalized 0..1, resized_raw (un-cropped) space
     "radius":  float,             # normalized to max(H, W)
     "connect": bool,              # also draw thick lines between consecutive pts
     "kind":    "spot" | "scratch",
     "source":  "auto" | "manual"}

See ``spec/dust-healing.md`` (§5, §9) for the full design.
"""
from __future__ import annotations

import numpy as np
import cv2

# --- tunable constants (spec §9.7) -----------------------------------------
SENSITIVITY_K = {"low": 6.0, "med": 4.5, "high": 3.0}
KERNEL_FRAC = 0.01           # morphology kernel ~ this fraction of the long side
                             # (detects dust up to ~half the kernel in radius)
MIN_KERNEL = 5
MAX_KERNEL = 51
MIN_AREA_FRAC = 0.0012       # min defect "radius" ~ this fraction of L (squared -> area)
MAX_AREA_FRAC = 0.05         # generous backstop; thin scratches are large-area but valid
                             # (real size gate is thickness, MAX_THICK_KERNELS below)
MAX_THICK_KERNELS = 1.0      # reject components thicker (inscribed radius) than this x kernel
                             # — a blob fatter than the morphology kernel isn't dust
ASPECT_SPOT = 2.0            # bbox aspect <= this AND solid enough -> spot, else scratch
SOLIDITY_SPOT = 0.6
SCRATCH_SAMPLE = 0.7         # grid spacing for scratch disc-sets, x the disc radius
HEAL_PAD_FRAC = 0.002        # mask dilation before inpaint ~ this fraction of L
TILE_EXTRA_PAD = 8           # extra px around each inpaint tile bbox
GRAIN = 0.3                  # grain strength (x local boundary std); 0 disables
MG_MAX_LEVELS = 3            # coarse-to-fine pyramid depth cap for the harmonic fill

# Brush radii for the manual heal tool, normalized to the image long side.
BRUSH_NORM = {"S": 0.004, "M": 0.008, "L": 0.016}

# Rec.601 luminance (matches ccr_processor._LUM_WEIGHTS).
_LUM_WEIGHTS = np.array([[0.299, 0.587, 0.114]], dtype=np.float32)


# --- small helpers ---------------------------------------------------------
def _odd(n: float) -> int:
    n = int(round(n))
    return n + 1 if n % 2 == 0 else n


def _luminance(img_u16: np.ndarray) -> np.ndarray:
    """HxWx3 uint16 RGB -> HxW float32 luminance."""
    return cv2.transform(img_u16.astype(np.float32), _LUM_WEIGHTS)


def _half_thickness(comp_u8: np.ndarray) -> float:
    """Inscribed-circle radius (half the local thickness) of a binary mask.
    A zero border is added first so a component that fills its bbox still has a
    background to measure against — otherwise distanceTransform returns FLT_MAX."""
    padded = cv2.copyMakeBorder(comp_u8, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dt = cv2.distanceTransform(padded, cv2.DIST_L2, 3)
    return float(dt.max())


def _grid_subsample(xs: np.ndarray, ys: np.ndarray, spacing: int) -> list:
    """Return indices keeping one point per ``spacing``-sized grid cell, so the
    kept points are at most ~sqrt(2)*spacing apart — discs of radius >= spacing
    then overlap and fully cover the component (spec §9.6)."""
    spacing = max(1, int(spacing))
    seen = set()
    keep = []
    gx = (xs // spacing).astype(np.int64)
    gy = (ys // spacing).astype(np.int64)
    for i in range(xs.shape[0]):
        key = (int(gx[i]), int(gy[i]))
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)
    return keep


def clean_strokes(strokes) -> list:
    """Drop malformed/empty strokes and coerce types — defensive against
    hand-edited or legacy catalogs (used by the catalog loader too)."""
    out = []
    for s in (strokes or []):
        if not isinstance(s, dict):
            continue
        pts = s.get("points")
        if not isinstance(pts, (list, tuple)) or not pts:
            continue
        clean_pts = []
        for p in pts:
            try:
                x, y = float(p[0]), float(p[1])
            except (TypeError, ValueError, IndexError):
                continue
            clean_pts.append([x, y])
        if not clean_pts:
            continue
        try:
            radius = float(s.get("radius", 0.006))
        except (TypeError, ValueError):
            radius = 0.006
        out.append({
            "points": clean_pts,
            "radius": max(1e-4, radius),
            "connect": bool(s.get("connect", False)),
            "kind": s.get("kind", "spot"),
            "source": s.get("source", "manual"),
        })
    return out


# --- detection -------------------------------------------------------------
def detect_dust(img_u16: np.ndarray, sensitivity: str = "med") -> list:
    """Detect dust spots and scratches on a converted positive (uint16 RGB).

    Returns a list of ``source="auto"`` stroke dicts (normalized coords).
    Deterministic; no RNG.
    """
    if img_u16 is None or getattr(img_u16, "ndim", 0) != 3:
        return []
    h, w = img_u16.shape[:2]
    L = max(h, w)

    k = int(np.clip(_odd(KERNEL_FRAC * L), MIN_KERNEL, MAX_KERNEL))
    lum = _luminance(img_u16)
    # Light pre-smooth so isolated 1-2 px film grain doesn't read as dust
    # (real specks are several px and survive a 3x3 blur).
    lum = cv2.GaussianBlur(lum, (3, 3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    tophat = cv2.morphologyEx(lum, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(lum, cv2.MORPH_BLACKHAT, kernel)
    residual = np.maximum(tophat, blackhat)

    # Robust threshold from the median absolute deviation (noise-adaptive).
    med = float(np.median(residual))
    mad = float(np.median(np.abs(residual - med)))
    spread = 1.4826 * mad + 1e-6          # ~ robust std
    k_sens = SENSITIVITY_K.get(sensitivity, SENSITIVITY_K["med"])
    thr = med + k_sens * spread
    mask = (residual > thr).astype(np.uint8)
    if int(mask.sum()) == 0:
        return []

    min_area = max(4, int(round((MIN_AREA_FRAC * L) ** 2)))
    max_area = max(min_area + 1, int(round(MAX_AREA_FRAC * h * w)))

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    strokes = []
    for lbl in range(1, num):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        x = int(stats[lbl, cv2.CC_STAT_LEFT])
        y = int(stats[lbl, cv2.CC_STAT_TOP])
        cw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        ch = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        comp = (labels[y:y + ch, x:x + cw] == lbl).astype(np.uint8)

        # Half-thickness (inscribed-circle radius). A component fatter than the
        # morphology kernel is a blob/real subject, not dust — reject it. This is
        # the true size gate (area would wrongly reject long thin scratches).
        half_thk = _half_thickness(comp)
        if half_thk > MAX_THICK_KERNELS * k:
            continue

        # Shape descriptors.
        aspect = max(cw, ch) / max(1, min(cw, ch))
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity = (area / hull_area) if hull_area > 0 else 1.0

        # Prominence gate: the peak must stand above its local surroundings.
        if not _is_prominent(residual, labels, lbl, x, y, cw, ch,
                             half_thk, k_sens * spread):
            continue

        is_spot = (aspect <= ASPECT_SPOT and solidity >= SOLIDITY_SPOT)
        if is_spot:
            cx, cy = float(centroids[lbl][0]), float(centroids[lbl][1])
            r_px = max(half_thk, float(np.sqrt(area / np.pi))) + 1.0
            strokes.append({
                "points": [[cx / w, cy / h]],
                "radius": r_px / L,
                "connect": False,
                "kind": "spot",
                "source": "auto",
            })
        else:
            r_px = max(1.0, half_thk) + 1.0
            spacing = max(1, int(round(SCRATCH_SAMPLE * r_px)))
            ys, xs = np.nonzero(comp)
            keep = _grid_subsample(xs, ys, spacing)
            pts = [[(x + int(xs[i])) / w, (y + int(ys[i])) / h] for i in keep]
            if not pts:
                pts = [[(x + cw / 2.0) / w, (y + ch / 2.0) / h]]
            strokes.append({
                "points": pts,
                "radius": r_px / L,
                "connect": False,
                "kind": "scratch",
                "source": "auto",
            })
    return strokes


def _is_prominent(residual, labels, lbl, x, y, cw, ch, half_thk, bar) -> bool:
    """True if the component's peak residual exceeds its surrounding-annulus
    median by more than ``bar`` (rejects detail sitting on bright texture)."""
    pad = int(min(50, max(2, round(3 * max(1.0, half_thk)))))
    y0 = max(0, y - pad)
    x0 = max(0, x - pad)
    y1 = min(residual.shape[0], y + ch + pad)
    x1 = min(residual.shape[1], x + cw + pad)
    sub_lbl = labels[y0:y1, x0:x1]
    sub_res = residual[y0:y1, x0:x1]
    comp = (sub_lbl == lbl)
    if not comp.any():
        return False
    comp_u8 = comp.astype(np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dil1 = cv2.dilate(comp_u8, ker, iterations=1)
    big = cv2.dilate(comp_u8, ker, iterations=pad)
    annulus = (big > 0) & (dil1 == 0)
    if not annulus.any():
        return True
    peak = float(sub_res[comp].max())
    ann_med = float(np.median(sub_res[annulus]))
    return (peak - ann_med) > bar


# --- rasterization + inpainting -------------------------------------------
def rasterize_strokes(strokes, h: int, w: int, pad_px: int = 0) -> np.ndarray:
    """Render strokes to a uint8 {0,255} mask of shape (h, w). Discs at every
    point (+ thick lines between consecutive points when ``connect``); finally
    dilated by ``pad_px`` so the inpaint covers each defect's soft edge."""
    mask = np.zeros((h, w), np.uint8)
    L = max(h, w)
    for st in (strokes or []):
        r = max(1, int(round(float(st.get("radius", 0.006)) * L)))
        pts = [(int(round(px * w)), int(round(py * h)))
               for px, py in st.get("points", [])]
        for (px, py) in pts:
            cv2.circle(mask, (px, py), r, 255, -1)
        if st.get("connect") and len(pts) >= 2:
            for a, b in zip(pts, pts[1:]):
                cv2.line(mask, a, b, 255, thickness=2 * r)
    if pad_px > 0 and mask.any():
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (2 * pad_px + 1, 2 * pad_px + 1))
        mask = cv2.dilate(mask, ker)
    return mask


def inpaint_dust(img_u16: np.ndarray, strokes, grain: float = GRAIN) -> np.ndarray:
    """Heal ``strokes`` on a converted positive of any resolution. Returns the
    input unchanged (same object) when there is nothing to heal."""
    if not strokes:
        return img_u16
    if img_u16.dtype != np.uint16:
        img_u16 = img_u16.astype(np.uint16)
    h, w = img_u16.shape[:2]
    L = max(h, w)
    pad = max(1, int(round(HEAL_PAD_FRAC * L)))
    mask = rasterize_strokes(strokes, h, w, pad_px=pad)
    if not mask.any():
        return img_u16

    out = img_u16.copy()
    # Tile by connected mask regions so cost scales with defect area, not image.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    tile_pad = TILE_EXTRA_PAD + pad
    for lbl in range(1, num):
        x = int(stats[lbl, cv2.CC_STAT_LEFT])
        y = int(stats[lbl, cv2.CC_STAT_TOP])
        cw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        ch = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        y0 = max(0, y - tile_pad)
        x0 = max(0, x - tile_pad)
        y1 = min(h, y + ch + tile_pad)
        x1 = min(w, x + cw + tile_pad)
        tile = out[y0:y1, x0:x1]                      # view into out
        tmask = (labels[y0:y1, x0:x1] == lbl).astype(np.uint8)
        if not tmask.any():
            continue
        filled = _inpaint_tile(tile, tmask, grain)
        hole = tmask.astype(bool)
        tile[hole] = np.clip(filled, 0, 65535).astype(np.uint16)[hole]
    return out


def _inpaint_tile(tile_u16: np.ndarray, hole_mask: np.ndarray,
                  grain: float) -> np.ndarray:
    """16-bit harmonic fill of one tile (float32 result, known pixels exact)."""
    f = tile_u16.astype(np.float32)
    hole = hole_mask.astype(bool)
    known = ~hole
    if not hole.any():
        return f
    if not known.any():
        f[:] = float(f.mean())
        return f
    filled = _harmonic_fill(f, hole)
    if grain > 0:
        _add_grain(filled, hole, f, grain)
    return filled


def _jacobi(f: np.ndarray, hole: np.ndarray, iters: int) -> np.ndarray:
    """Relax ``f`` toward the harmonic solution, keeping known pixels fixed."""
    if iters <= 0:
        return f
    hole3 = np.repeat(hole[..., None], f.shape[2], axis=2)
    for _ in range(iters):
        blur = cv2.blur(f, (3, 3))
        f[hole3] = blur[hole3]
    return f


def _harmonic_fill(f: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Solve Laplace on the hole with Dirichlet BC via a coarse-to-fine seed,
    so wide holes converge in few fine iterations (spec §9.2). Reproduces a
    local linear gradient exactly, so thin defects on a sky gradient heal with
    no halo."""
    h, w = hole.shape
    dt = cv2.distanceTransform(hole.astype(np.uint8), cv2.DIST_L2, 3)
    thick = 2.0 * float(dt.max())

    # Mean-seed the hole interior (known pixels stay exact).
    known = ~hole
    for c in range(f.shape[2]):
        ch = f[..., c]
        ch[hole] = float(ch[known].mean())

    # Decide pyramid depth.
    levels = 0
    while (thick / (2 ** levels)) > 4 and (min(h, w) >> (levels + 1)) >= 8 \
            and levels < MG_MAX_LEVELS:
        levels += 1

    pyr_f = [f]
    pyr_hole = [hole]
    for _ in range(levels):
        pyr_f.append(cv2.pyrDown(pyr_f[-1]))
        ph, pw = pyr_f[-1].shape[:2]
        hm = cv2.resize(pyr_hole[-1].astype(np.uint8), (pw, ph),
                        interpolation=cv2.INTER_AREA)
        pyr_hole.append(hm > 0)

    for lvl in range(levels, -1, -1):
        fl = pyr_f[lvl]
        hl = pyr_hole[lvl]
        if lvl < levels:
            up = cv2.resize(pyr_f[lvl + 1], (fl.shape[1], fl.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
            h3 = np.repeat(hl[..., None], fl.shape[2], axis=2)
            fl[h3] = up[h3]
        local_thick = thick / (2 ** lvl)
        iters = int(np.clip(round(1.5 * local_thick), 8, 64))
        pyr_f[lvl] = _jacobi(fl, hl, iters)
    return pyr_f[0]


def _add_grain(filled: np.ndarray, hole: np.ndarray, original: np.ndarray,
               grain: float) -> None:
    """Add deterministic zero-mean noise to the healed pixels, scaled to the
    local boundary-ring std, so the patch isn't a tell-tale smooth blob."""
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))   # ~6 px ring
    dil = cv2.dilate(hole.astype(np.uint8), ker)
    ring = (dil > 0) & (~hole)
    if not ring.any():
        return
    seed = int(filled.mean()) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)
    idx = np.nonzero(hole)
    n = idx[0].size
    for c in range(filled.shape[2]):
        sd = float(original[..., c][ring].std())
        if sd <= 0:
            continue
        noise = rng.normal(0.0, grain * sd, size=n).astype(np.float32)
        filled[..., c][idx] += noise
