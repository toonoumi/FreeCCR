"""Classical auto-white-balance (AWB) illuminant estimation.

Learning-free estimators over the converted base (spec/auto-white-balance.md).
Each algorithm returns the RGB triple that *should* be neutral — exactly what a
perfect grey-spot pick would sample — so the result feeds
compute_neutral_balance unchanged and lands on the R/G/B Balance sliders
through the same inverse the WB eyedropper uses. See spec/channel-balance.md.
"""

import numpy as np

from core.ccr_processor import (MIN_CONTENT_FRACTION, WS_B, _WS_INV_WIDTH,
                                apply_crop_to_image, encode_window)

# (id, UI label) — ordered as shown in the Settings dropdown.
AWB_ALGORITHMS = [
    ("gray_world", "Gray World"),
    ("white_patch", "White Patch"),
    ("shades_of_gray", "Shades of Gray"),
    ("gray_edge", "Gray Edge"),
]
AWB_DEFAULT = "gray_world"

# An uncropped film scan always contains pure black AND pure white that is not
# scene content: the holder masks the film to pure black in the scan (→ clipped
# white once inverted) and the clear film base / sprocket holes are the scan's
# maximum (→ crushed black once inverted). Neither carries a usable cast, and
# both are big enough to hijack every estimator. So AWB reads the MIDTONES plus
# a slice of the shadows and highlights, and nothing else:
#   1. per-channel gate — no channel crushed or blown, so the ratios are real
#   2. luminance band  — the tonal region the WB decision is actually made from
AWB_LO = 0.06          # per-channel floor: clear-film black / crushed channel
AWB_HI = 0.94          # per-channel ceiling: holder white / blown channel
AWB_TONE_LO = 0.15     # luminance band: midtones + a slice of the shadows...
AWB_TONE_HI = 0.85     # ...and a slice of the highlights
AWB_EPS = 1e-6
_WP_PERCENTILE = 99.0  # white_patch: robust max-RGB
_MINK_P = 6.0          # Minkowski norm for shades_of_gray / gray_edge
_GE_SIGMA = 1.0        # gray_edge pre-smoothing
_GE_ERODE = 5          # gray_edge: mask shrink (px kernel) around rejected pixels
_GE_TOP_FRACTION = 0.2 # gray_edge (closed loop): fraction of edge pixels kept
# Rec.601 luma over RGB — the convention compute_auto_gain_offset uses.
_LUMA = (0.299, 0.587, 0.114)


def _luminance(flat):
    """Luma of an (N,3) RGB float array."""
    return (_LUMA[0] * flat[:, 0] + _LUMA[1] * flat[:, 1] + _LUMA[2] * flat[:, 2])


def _gray_edge_estimate(d, valid_mask):
    """van de Weijer gray-edge: Minkowski p-mean of the smoothed per-channel
    gradient magnitude, over pixels whose source value is in-bound."""
    import cv2
    sm = cv2.GaussianBlur(d, (0, 0), _GE_SIGMA)
    gy, gx = np.gradient(sm, axis=(0, 1))
    mag = np.sqrt(gx * gx + gy * gy)
    # The blur + gradient stencil reaches a couple of pixels, so a sample that
    # merely sits NEXT to a rejected pixel still carries that pixel's edge — the
    # holder border is the strongest edge in an uncropped scan and would
    # otherwise dominate the p-mean. Shrink the mask by that reach.
    kept = cv2.erode(valid_mask.astype(np.uint8),
                     np.ones((_GE_ERODE, _GE_ERODE), np.uint8)).astype(bool)
    m = mag.reshape(-1, 3)[kept.reshape(-1)]
    if m.shape[0] == 0:
        return None
    return np.mean(m ** _MINK_P, axis=0) ** (1.0 / _MINK_P)


def gray_edge_pixel_mask(base_u16, ws_windowed=False,
                         top_fraction=_GE_TOP_FRACTION):
    """Boolean mask of the strongest-edge pixels of `base_u16`.

    gray_edge is the one estimator whose output is NOT a per-channel value
    statistic — it is gradient ENERGY, and energy falls as a channel is lifted
    (the Balance curve compresses above its node). Driving those three energies
    to equality therefore pushes colour the WRONG way. Under the closed loop the
    estimator is used for what it is actually saying instead — *these* are the
    pixels that carry the illuminant — and their rendered mean is what gets
    neutralised. Same premise as van de Weijer, expressed as a pixel selection.

    Computed once on the BASE so the objective is fixed while the loop runs; a
    mask that moved between iterations would not converge. Returns None when
    too few pixels qualify."""
    import cv2
    d = base_u16.astype(np.float32)
    if ws_windowed:
        d = (d - WS_B) * _WS_INV_WIDTH
    else:
        d = d * (1.0 / 65535.0)
    lum = _luminance(d.reshape(-1, 3)).reshape(d.shape[:2])
    valid = (np.all((d >= AWB_LO) & (d <= AWB_HI), axis=2)
             & (lum >= AWB_TONE_LO) & (lum <= AWB_TONE_HI))
    sm = cv2.GaussianBlur(d, (0, 0), _GE_SIGMA)
    gy, gx = np.gradient(sm, axis=(0, 1))
    mag = np.sqrt(gx * gx + gy * gy).sum(axis=2)
    # Same erosion as _gray_edge_estimate: the blur + gradient stencil reaches a
    # couple of pixels, so a sample merely NEXT to a rejected pixel still carries
    # that pixel's edge (the holder border dominates otherwise).
    kept = cv2.erode(valid.astype(np.uint8),
                     np.ones((_GE_ERODE, _GE_ERODE), np.uint8)).astype(bool)
    if kept.sum() < MIN_CONTENT_FRACTION * kept.size:
        return None
    cut = float(np.percentile(mag[kept], 100.0 * (1.0 - top_fraction)))
    mask = kept & (mag >= cut)
    if mask.sum() < MIN_CONTENT_FRACTION * mask.size:
        return None
    return mask


def estimate_neutral_rgb(base_u16, ws_windowed=False, algorithm=AWB_DEFAULT):
    """Estimated neutral (cast) RGB triple of a converted base, or None when
    there is not enough usable content. The scale is arbitrary —
    Values are NORMALISED display units in [0,1] — the domain
    compute_neutral_balance needs (a tone-weighted control needs the absolute
    tone, not just the ratio). Index 0=R, 1=G, 2=B.
    Only midtone pixels vote (AWB_LO/AWB_HI + AWB_TONE_LO/AWB_TONE_HI), so the
    pure black and pure white an uncropped film scan always carries — clear
    film base, sprocket holes, the holder surround — cannot reach any
    estimator, white_patch included.
    Unknown algorithm ids fall back to gray_world (forward-compat settings)."""
    if base_u16 is None or getattr(base_u16, "size", 0) == 0:
        return None
    d = base_u16.astype(np.float32)
    if ws_windowed:
        d -= np.float32(WS_B)
        d *= np.float32(_WS_INV_WIDTH)     # de-window; headroom masked below
    else:
        d *= np.float32(1.0 / 65535.0)
    flat = d.reshape(-1, 3)
    lum = _luminance(flat)
    valid = (np.all((flat >= AWB_LO) & (flat <= AWB_HI), axis=1)
             & (lum >= AWB_TONE_LO) & (lum <= AWB_TONE_HI))
    if valid.sum() < MIN_CONTENT_FRACTION * flat.shape[0]:
        return None
    if algorithm == "white_patch":
        est = np.percentile(flat[valid], _WP_PERCENTILE, axis=0)
    elif algorithm == "shades_of_gray":
        est = np.mean(flat[valid] ** _MINK_P, axis=0) ** (1.0 / _MINK_P)
    elif algorithm == "gray_edge":
        est = _gray_edge_estimate(d, valid.reshape(d.shape[:2]))
    else:                                  # gray_world + unknown-id fallback
        est = np.mean(flat[valid], axis=0)
    if est is None or not np.all(np.isfinite(est)) or np.any(est <= AWB_EPS):
        return None
    return float(est[0]), float(est[1]), float(est[2])


# The whole frame is rendered on every loop iteration, so it is first reduced to
# this long side. Small enough to render in a few ms, large enough that the
# estimators still see real structure — gray_edge in particular needs SPATIAL
# neighbours, which a scattered pixel sample would destroy.
AWB_SOLVE_LONG_SIDE = 256


def downscale_for_solve(base_u16, long_side=AWB_SOLVE_LONG_SIDE):
    """A small, spatially intact stand-in for the whole frame.

    Area-averaging is safe on a WINDOWED base: the window encode is affine, so
    the mean of codes is the code of the mean display value."""
    import cv2
    h, w = base_u16.shape[:2]
    scale = float(long_side) / float(max(h, w))
    if scale >= 1.0:
        return base_u16
    return cv2.resize(base_u16, (max(1, int(round(w * scale))),
                                 max(1, int(round(h * scale)))),
                      interpolation=cv2.INTER_AREA)


def compute_awb_balance(ccr_image, algorithm=None):
    """(balance_r, balance_g, balance_b) slider ints that neutralize the image's
    overall cast, or None when there is not enough usable content.

    This is a REGRESSION OVER THE WHOLE FRAME, not a solve at one estimated
    pixel. A downscaled copy of the (cropped) frame is rendered through the real
    adjustment pipeline on every iteration, the illuminant estimator runs on
    THAT — the rendered result, the thing the user actually sees — and the loop
    drives its estimate to grey. Red is the anchor, blue then green are solved
    by bisection on the measured output.

    Solving at a single estimated pixel (the previous approach) was fragile:
    the estimate is measured on the BASE, but the tone it lands on after Channel
    Levels and Auto Gain may be up where a low-anchored Balance node has almost
    no authority, so the solve produced tiny or zero values and the button
    looked dead. Optimising the rendered frame's own estimate has no such blind
    spot — every tone in the image contributes.

    Runs on the converted base (resized_raw), so the result does not depend on
    the current Balance values (idempotent). Crop-aware: with a crop set, only
    the kept region drives the estimate (a rotated crop's black corner fill sits
    below AWB_LO, so the mask discards it). Uses the backend-selected algorithm
    when none is given. See spec/channel-balance.md and spec/auto-white-balance.md.
    """
    if ccr_image is None or getattr(ccr_image, "resized_raw", None) is None:
        return None
    if algorithm is None:
        from core.ccr_backend import ccr_backend   # deferred: circular at load
        algorithm = getattr(ccr_backend, "awb_algorithm", AWB_DEFAULT)
    base = apply_crop_to_image(ccr_image.resized_raw,
                               getattr(ccr_image, "crop_rect", None),
                               getattr(ccr_image, "crop_angle", 0.0) or 0.0)
    ws = bool(getattr(ccr_image, "_ws_windowed", False))
    sample = downscale_for_solve(base)
    # Bail out the same way as before when the frame carries too little usable
    # content for any estimator to speak — checked on the BASE, before any
    # rendering, so a hopeless frame costs nothing.
    if estimate_neutral_rgb(sample, ws, algorithm) is None:
        return None

    if algorithm == "gray_edge":
        # Not a value statistic — see gray_edge_pixel_mask. Use it to pick the
        # illuminant-carrying pixels, then neutralise THEIR rendered mean.
        mask = gray_edge_pixel_mask(sample, ws)
        if mask is None:
            return None
        flat_mask = mask.reshape(-1)

        def combine(rendered):
            sel = rendered.reshape(-1, 3)[flat_mask]
            return None if sel.size == 0 else sel.mean(axis=0)
    else:
        def combine(rendered):
            """The cast of the RENDERED frame: what the loop drives to grey. The
            render is a normal full-range display image, never a windowed base.

            estimate_neutral_rgb reports in NORMALISED [0,1]; the solver works in
            the render's 0–65535 scale, so scale it. Returning None (a render with
            too little in-gate content, e.g. blown by a heavy gain) lets the loop
            fall back to the plain mean rather than stalling."""
            est = estimate_neutral_rgb(rendered, False, algorithm)
            return None if est is None else np.asarray(est) * 65535.0

    return ccr_image.solve_neutral_balance(sample, combine=combine)
