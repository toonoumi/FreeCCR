"""
Auto White Balance (AWB) — learning-based colour-cast removal.

Wraps a learning-based white-balance network (``net_awb.onnx``): the net takes a
display-referred RGB image and returns a white-balanced version of it; we derive
a per-channel linear gain from the ratio of input vs. corrected channel means and
hand that back to the caller, which multiplies it onto the converted positive
*before* the manual sliders.

This module is deliberately self-contained and imports ``onnxruntime`` ONLY
inside functions, so importing it (and anything that uses it) never fails when
onnxruntime is absent — AWB simply reports itself unavailable and the rest of the
app is unaffected. Same defensive pattern as ``core.dust_detect``.

The model is BUNDLED with FreeCCR (``src/models/net_awb.onnx`` in dev, ``models/``
next to the executable in the Nuitka build); see ``spec/auto-white-balance.md``.
"""
import os
import sys
import threading

import cv2
import numpy as np

MODEL_FILENAME = "net_awb.onnx"

# Inference + gain tuning ----------------------------------------------------
AWB_INFER_SIZE = 256       # long side the net runs at (speed vs. accuracy)
AWB_SIZE_MULTIPLE = 16     # both dims rounded to a multiple of this (model req.)
AWB_GAIN_MIN = 0.3         # clamp each per-channel gain to guard against
AWB_GAIN_MAX = 3.0         # pathological corrections

_session = None            # cached ort.InferenceSession
_session_path = None       # model path the cached session was built from
_session_lock = threading.Lock()


def model_path() -> str:
    """Resolve the bundled model. Nuitka build: ``models/`` next to the exe;
    dev: ``src/models/``; finally ``%APPDATA%/FreeCCR/models`` (parity with the
    dust detector, lets a user drop the file in manually)."""
    if getattr(sys, "frozen", False):
        cand = os.path.join(os.path.dirname(sys.executable), "models", MODEL_FILENAME)
        if os.path.exists(cand):
            return cand
    dev = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models",
                                       MODEL_FILENAME))
    if os.path.exists(dev):
        return dev
    base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "FreeCCR", "models", MODEL_FILENAME)


def is_model_present() -> bool:
    """True when the bundled model file exists and is non-empty."""
    try:
        return os.path.getsize(model_path()) > 0
    except OSError:
        return False


def is_available() -> bool:
    """True when AWB can run: onnxruntime importable AND the model present."""
    if not is_model_present():
        return False
    try:
        import onnxruntime  # noqa: F401  (late import — never at module level)
        return True
    except Exception:
        return False


def availability_reason() -> str:
    """Human-readable reason AWB can't run, or '' when available. Surfaced in
    the panel tooltip so the user knows what's wrong."""
    try:
        import onnxruntime  # noqa: F401
    except Exception as e:
        return ("Auto WB needs the 'onnxruntime' package, which isn't importable "
                f"({type(e).__name__}). Run `pip install -r requirements.txt`, "
                "then restart FreeCCR.")
    if not is_model_present():
        return (f"Auto WB model '{MODEL_FILENAME}' is missing. Expected at "
                f"{model_path()}.")
    return ""


def _get_session():
    """Lazily build (and cache) the CPU ONNX inference session."""
    global _session, _session_path
    with _session_lock:
        path = model_path()
        if _session is not None and _session_path == path:
            return _session
        if not is_model_present():
            raise FileNotFoundError(f"AWB model not found at {path}")
        import onnxruntime as ort  # late import
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        _session = sess
        _session_path = path
        return sess


# --- sRGB transfer (the net was trained on display-referred sRGB) -----------
def _srgb_encode(x: np.ndarray) -> np.ndarray:
    """linear [0,1] -> sRGB [0,1]."""
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(x <= 0.0031308, 12.92 * x, (1 + a) * np.power(x, 1 / 2.4) - a)


def _srgb_decode(x: np.ndarray) -> np.ndarray:
    """sRGB [0,1] -> linear [0,1]."""
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, np.power((x + a) / (1 + a), 2.4))


def _infer_size(h: int, w: int) -> tuple:
    """Inference resolution: long side ~AWB_INFER_SIZE (never upscaled), both
    dims rounded to a multiple of AWB_SIZE_MULTIPLE (model requirement)."""
    long_side = max(1, max(h, w))
    scale = min(1.0, AWB_INFER_SIZE / long_side)
    dh = max(AWB_SIZE_MULTIPLE,
             int(round(h * scale / AWB_SIZE_MULTIPLE)) * AWB_SIZE_MULTIPLE)
    dw = max(AWB_SIZE_MULTIPLE,
             int(round(w * scale / AWB_SIZE_MULTIPLE)) * AWB_SIZE_MULTIPLE)
    return dh, dw


def compute_gains(positive_rgb16: np.ndarray):
    """Estimate a per-channel linear white-balance gain for a converted positive.

    ``positive_rgb16`` is the near-linear 16-bit RGB positive (H, W, 3). Returns
    ``(r_gain, g_gain, b_gain)`` floats — luminance-normalised so the correction
    shifts only colour, not overall exposure — or ``None`` if onnxruntime/model
    is unavailable or inference fails (caller then leaves the image untouched).

    Pipeline (see spec/auto-white-balance.md §6.2): sRGB-encode -> net -> decode
    both back to linear -> per-channel mean ratio -> luminance-normalise -> clamp.
    """
    if positive_rgb16 is None or positive_rgb16.ndim != 3 or positive_rgb16.shape[2] < 3:
        return None
    try:
        sess = _get_session()
    except Exception:
        return None
    try:
        lin = positive_rgb16[..., :3].astype(np.float32) / 65535.0
        h, w = lin.shape[:2]
        dh, dw = _infer_size(h, w)
        small_lin = cv2.resize(lin, (dw, dh), interpolation=cv2.INTER_AREA)
        # The net expects display-referred sRGB in [0,1], NCHW.
        enc = _srgb_encode(small_lin).astype(np.float32)
        inp = np.ascontiguousarray(enc.transpose(2, 0, 1)[None])
        name = sess.get_inputs()[0].name
        out = sess.run(None, {name: inp})[0]
        out = np.asarray(out, dtype=np.float32)
        if out.ndim == 4:
            out = out[0]
        out_enc = np.clip(out.transpose(1, 2, 0), 0.0, 1.0)
        if out_enc.shape[:2] != (dh, dw):
            out_enc = cv2.resize(out_enc, (dw, dh), interpolation=cv2.INTER_LINEAR)
        # Measure the gain in LINEAR space (where it is applied).
        in_lin = _srgb_decode(enc)
        out_lin = _srgb_decode(out_enc)
        in_mean = np.maximum(in_lin.reshape(-1, 3).mean(axis=0), 1e-6)
        out_mean = out_lin.reshape(-1, 3).mean(axis=0)
        gains = out_mean / in_mean
        lum = 0.299 * gains[0] + 0.587 * gains[1] + 0.114 * gains[2]
        if not np.isfinite(lum) or lum <= 1e-6:
            return None
        gains = gains / lum  # luminance-preserving: colour shift only
        gains = np.clip(gains, AWB_GAIN_MIN, AWB_GAIN_MAX)
        if not np.all(np.isfinite(gains)):
            return None
        return (float(gains[0]), float(gains[1]), float(gains[2]))
    except Exception as e:
        print(f"AWB gain computation failed: {e}")
        return None


def apply_gains(image_rgb16: np.ndarray, gains) -> np.ndarray:
    """Multiply a 16-bit RGB image by per-channel ``gains`` (r, g, b), clipping
    to [0, 65535] and preserving dtype. Identity gains (1,1,1) are a no-op.
    Resolution-independent — the same gains apply at any resolution."""
    if gains is None:
        return image_rgb16
    gr, gg, gb = float(gains[0]), float(gains[1]), float(gains[2])
    if gr == 1.0 and gg == 1.0 and gb == 1.0:
        return image_rgb16
    out = image_rgb16.astype(np.float32)
    out[..., 0] *= gr
    out[..., 1] *= gg
    out[..., 2] *= gb
    np.clip(out, 0, 65535, out=out)
    return out.astype(image_rgb16.dtype)
