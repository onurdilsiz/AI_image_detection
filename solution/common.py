#!/usr/bin/env python3
"""Shared helpers for the Task 1.2 modeling pipeline.

Imported by ``prepare.py``, ``train.py`` and ``predict.py`` (the grader runs all
scripts from the ``solution/`` folder, so a plain module import works).

Everything here is CPU-only, internet-free and deterministic. The two design
constraints that the rest of the pipeline relies on:

* **Square-resize to 128x128** neutralises the Task 1.1 "size shortcut"
  (AI images are 100% square 320px / ~25 KB; real images are larger and
  non-square). After resizing, that trivial signal is gone and the model must
  learn content.
* **Leakage-safe engineered features**: we deliberately compute *only*
  content / frequency / noise statistics and NEVER width / height / file-bytes /
  aspect. Those would let a classifier relearn the shortcut and then fail on the
  hidden holdout.
"""

from __future__ import annotations

import io
import os
import random
import time

import numpy as np

# ----------------------------------------------------------------------------
# Constants / paths
# ----------------------------------------------------------------------------

SEED = 0
RESIZE = 128            # fixed square side that neutralises the size shortcut
CNN_RES = 64            # CNN trains on a 64px view of the 128px cache (4x cheaper
                        # per step -> many more epochs fit the 1800s budget)
CNN_K = 32              # Appendix-B base channel width (fallbacks 24/16)
MAX_FPR = 0.20          # hard constraint: FPR_real must stay <= 20%
CALIB_TARGET_FPR = 0.17  # calibrate below the 20% cap to absorb the
                         # calibration->validation FPR gap (keeps the strong
                         # ensemble's validation FPR safely under the gate)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
ART = os.path.join(HERE, "artifacts")
PREP = os.path.join(ART, "prepared")
TASK02 = os.path.join(ART, "task02")
TASK03 = os.path.join(ART, "task03")
CLEAN_MANIFEST = os.path.join(ART, "clean", "clean_manifest.parquet")

# Labeled splits that prepare.py caches (predict is intentionally excluded).
LABELED_SPLITS = [
    "train",
    "calibration",
    "validation",
    "validation_augmented",
    "calibration_augmented",
]

BATCH_SIZE = 256  # pyarrow streaming batch size


# ----------------------------------------------------------------------------
# Label mapping (real == source_class 0; AI source classes 1..5 -> 1)
# ----------------------------------------------------------------------------

def to_label(source_class: int) -> int:
    return 0 if int(source_class) == 0 else 1


# ----------------------------------------------------------------------------
# Deadline helper (copied from clean.py for a consistent timeout contract)
# ----------------------------------------------------------------------------

class Deadline:
    """Monotonic wall-clock budget with a safety factor.

    The grader kills the process at ``timeout_seconds``; we stop the expensive
    work a little early so outputs / checkpoints are always written.
    """

    def __init__(self, timeout_seconds: float, safety: float = 0.9):
        self.start = time.monotonic()
        self.budget = max(1.0, float(timeout_seconds) * safety)

    def remaining(self) -> float:
        return self.budget - (time.monotonic() - self.start)

    def expired(self) -> bool:
        return self.remaining() <= 0.0

    def elapsed(self) -> float:
        return time.monotonic() - self.start


# ----------------------------------------------------------------------------
# Determinism / threading (matches Appendix C, --cpus 8 grading)
# ----------------------------------------------------------------------------

def set_determinism(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


def set_threads() -> None:
    try:
        import torch
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        torch.set_num_interop_threads(1)
    except Exception:
        pass


# ----------------------------------------------------------------------------
# Decode / resize
# ----------------------------------------------------------------------------

def decode_resize(raw: bytes, size: int = RESIZE):
    """Decode raw image bytes -> uint8 ``[size, size, 3]`` RGB, or None on error.

    The square resize is the shortcut-neutralising transform: every image,
    regardless of original resolution / aspect, becomes the same square grid.
    """
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        im = im.resize((size, size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)
        if arr.shape != (size, size, 3):
            return None
        return arr
    except Exception:
        return None


def downsample_u8(batch_u8: np.ndarray, src: int = RESIZE, dst: int = CNN_RES):
    """Exact block-mean downsample of a uint8 image batch -> uint8.

    ``batch_u8`` is ``[N, src, src, 3]`` (or a single ``[src, src, 3]`` image).
    Reshapes into ``factor x factor`` blocks and averages them. Deterministic,
    no PIL/scipy. Used to give the CNN a cheaper 64px view of the 128px cache so
    far more training epochs fit the budget; the architecture is unchanged
    (AdaptiveAvgPool2d makes it resolution-agnostic).
    """
    if src % dst != 0:
        raise ValueError(f"src {src} not divisible by dst {dst}")
    factor = src // dst
    single = (batch_u8.ndim == 3)
    arr = batch_u8[None] if single else batch_u8
    n = arr.shape[0]
    x = arr.astype(np.float32).reshape(n, dst, factor, dst, factor, 3)
    x = x.mean(axis=(2, 4))
    out = np.clip(x + 0.5, 0, 255).astype(np.uint8)
    return out[0] if single else out


# ----------------------------------------------------------------------------
# Robustness augmentations (Task 1.3): scaled / compressed / blurred / noisy
# ----------------------------------------------------------------------------

def augment_u8(img_u8: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply ONE random robustness augmentation to a uint8 [H,W,3] image.

    Mirrors the perturbations named in the brief (scaled, compressed, blurred)
    plus mild additive noise. Each call uses ``rng`` so a fixed seed makes the
    augmented training set reproducible. Output is the same shape/dtype.
    """
    from PIL import Image, ImageFilter
    choice = int(rng.integers(0, 4))
    im = Image.fromarray(img_u8)
    if choice == 0:  # JPEG recompression artifacts
        q = int(rng.integers(35, 86))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        out = np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)
    elif choice == 1:  # Gaussian blur
        radius = float(rng.uniform(0.5, 1.5))
        out = np.asarray(im.filter(ImageFilter.GaussianBlur(radius=radius)),
                         dtype=np.uint8)
    elif choice == 2:  # downscale then upscale (resolution loss)
        f = float(rng.uniform(0.5, 0.85))
        s = max(8, int(round(img_u8.shape[0] * f)))
        small = im.resize((s, s), Image.BILINEAR)
        out = np.asarray(small.resize(im.size, Image.BILINEAR), dtype=np.uint8)
    else:  # additive Gaussian noise
        sigma = float(rng.uniform(3.0, 12.0))
        noisy = img_u8.astype(np.float32) + rng.normal(
            0.0, sigma, img_u8.shape).astype(np.float32)
        out = np.clip(noisy, 0, 255).astype(np.uint8)
    if out.shape != img_u8.shape:
        out = np.asarray(Image.fromarray(out).resize(
            (img_u8.shape[1], img_u8.shape[0]), Image.BILINEAR), dtype=np.uint8)
    return out


def augment_extended_u8(img_u8: np.ndarray, rng: np.random.Generator,
                         mode: str = "random", severity: float = 0.5) -> np.ndarray:
    """Extended augmentations for robustness evaluation.

    mode: one of {"random", "jpeg", "blur", "downscale", "noise",
    "color_jitter", "crop", "rotate", "saltpepper", "combined"}.
    severity: [0,1] scales the perturbation strength.
    """
    from PIL import Image, ImageFilter, ImageEnhance

    im = Image.fromarray(img_u8)
    h, w = img_u8.shape[0], img_u8.shape[1]
    m = mode
    if m == "random":
        choices = ["jpeg", "blur", "downscale", "noise", "color_jitter",
                   "crop", "rotate", "saltpepper"]
        m = choices[int(rng.integers(0, len(choices)))]

    def _clip_arr(a):
        return np.clip(a, 0, 255).astype(np.uint8)

    if m == "jpeg":
        qmin, qmax = 20, 95
        q = int(qmax - (qmax - qmin) * severity)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        out = np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)
    elif m == "blur":
        radius = 0.5 + 4.5 * severity
        out = np.asarray(im.filter(ImageFilter.GaussianBlur(radius=radius)),
                         dtype=np.uint8)
    elif m == "downscale":
        f = 1.0 - 0.9 * severity  # 1.0 -> 0.1
        s = max(8, int(round(h * f)))
        small = im.resize((s, s), Image.BILINEAR)
        out = np.asarray(small.resize((w, h), Image.BILINEAR), dtype=np.uint8)
    elif m == "noise":
        sigma = 1.0 + 50.0 * severity
        noisy = img_u8.astype(np.float32) + rng.normal(0.0, sigma, img_u8.shape)
        out = _clip_arr(noisy)
    elif m == "color_jitter":
        # brightness, contrast, saturation
        b = 1.0 + (rng.normal() * 0.3) * severity
        c = 1.0 + (rng.normal() * 0.3) * severity
        s = 1.0 + (rng.normal() * 0.3) * severity
        out_im = ImageEnhance.Brightness(im).enhance(b)
        out_im = ImageEnhance.Contrast(out_im).enhance(c)
        out_im = ImageEnhance.Color(out_im).enhance(s)
        out = np.asarray(out_im, dtype=np.uint8)
    elif m == "crop":
        p = 0.0 + 0.4 * severity
        crop_h = int(round(h * (1.0 - p)))
        crop_w = int(round(w * (1.0 - p)))
        top = int(rng.integers(0, h - crop_h + 1))
        left = int(rng.integers(0, w - crop_w + 1))
        cropped = im.crop((left, top, left + crop_w, top + crop_h))
        out = np.asarray(cropped.resize((w, h), Image.BILINEAR), dtype=np.uint8)
    elif m == "rotate":
        angle = (-30.0 + 60.0 * severity) * (1.0 if rng.random() < 0.5 else -1.0)
        out = np.asarray(im.rotate(angle, resample=Image.BILINEAR, expand=False),
                         dtype=np.uint8)
    elif m == "saltpepper":
        amount = 0.001 + 0.09 * severity
        arr = img_u8.copy().astype(np.float32)
        n = int(arr.size * amount)
        if n > 0:
            idx = rng.integers(0, arr.shape[0] * arr.shape[1], size=n)
            ys = idx // arr.shape[1]
            xs = idx % arr.shape[1]
            for y, x in zip(ys, xs):
                arr[y, x, int(rng.integers(0, 3))] = 255 if rng.random() < 0.5 else 0
        out = _clip_arr(arr)
    else:  # combined: apply two random ops
        a = augment_extended_u8(img_u8, rng, mode="random", severity=severity)
        out = augment_extended_u8(a, rng, mode="random", severity=min(1.0, severity * 1.2))

    if out.shape != img_u8.shape:
        out = np.asarray(Image.fromarray(out).resize((w, h), Image.BILINEAR),
                         dtype=np.uint8)
    return out


# ----------------------------------------------------------------------------
# Leakage-safe engineered features (content / frequency / noise only)
# ----------------------------------------------------------------------------

def _box_blur3(gray: np.ndarray) -> np.ndarray:
    """3x3 mean blur via padded shift-and-add (no scipy dependency)."""
    p = np.pad(gray, 1, mode="edge")
    acc = np.zeros_like(gray, dtype=np.float32)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            acc += p[dy:dy + gray.shape[0], dx:dx + gray.shape[1]]
    return acc / 9.0


def _moments(x: np.ndarray):
    """Return (mean, std, skew, excess-kurtosis) of a flat array."""
    x = x.ravel().astype(np.float64)
    mu = x.mean()
    sd = x.std()
    if sd < 1e-8:
        return float(mu), 0.0, 0.0, 0.0
    z = (x - mu) / sd
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean() - 3.0)
    return float(mu), float(sd), skew, kurt


def _residual_spectrum(gray: np.ndarray, nbins: int = 16):
    """Residual-spectrum fingerprint of a [H,W] gray image in [0,1].

    Generative up-samplers (GAN/diffusion) leave periodic high-frequency
    artifacts; real images have a smooth, natural spectral fall-off. We isolate
    that signal by taking the FFT of the *noise residual* (gray minus a 3x3
    blur, a cheap high-pass), then summarise it in a rotation-invariant way:

      * 16-bin radially-averaged log-power "reduced spectrum"
      * peakiness of the outer (high-freq) annulus -> periodic-spike detector
      * std of the outer annulus and kurtosis of the whole log-power map

    All inputs are post-square-resize pixels only (no width/height/bytes), so
    this stays leakage-safe w.r.t. the Task 1.1 size shortcut.
    """
    resid = gray - _box_blur3(gray)
    f = np.fft.fftshift(np.fft.fft2(resid))
    power = f.real ** 2 + f.imag ** 2
    logp = np.log1p(power)

    h, w = gray.shape
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rn = radius / (radius.max() + 1e-8)

    edges = np.linspace(0.0, 1.0, nbins + 1)
    radial = np.zeros(nbins, dtype=np.float32)
    for i in range(nbins):
        m = (rn >= edges[i]) & (rn < edges[i + 1])
        if m.any():
            radial[i] = float(logp[m].mean())

    outer = logp[rn > 0.6]
    if outer.size:
        omu = float(outer.mean()); osd = float(outer.std())
        peakz = float((outer.max() - omu) / (osd + 1e-8))
    else:
        osd, peakz = 0.0, 0.0
    _, _, _, lk = _moments(logp)
    return radial, peakz, osd, lk


def _channel_hf_stats(ch: np.ndarray):
    """Compact high-frequency residual stats for one channel in [0,1].

    Returns (outer_mean_logpow, outer_peakz) of the FFT of the channel's noise
    residual, where "outer" is the high-frequency annulus (normalised radius
    > 0.6). Captures per-channel / chroma generation artifacts that the
    grayscale residual spectrum can wash out. Leakage-safe (post-resize pixels).
    """
    resid = ch - _box_blur3(ch)
    f = np.fft.fftshift(np.fft.fft2(resid))
    logp = np.log1p(f.real ** 2 + f.imag ** 2)
    h, w = ch.shape
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2)
    rn = radius / (radius.max() + 1e-8)
    outer = logp[rn > 0.6]
    if outer.size:
        omu = float(outer.mean()); osd = float(outer.std())
        return omu, float((outer.max() - omu) / (osd + 1e-8))
    return 0.0, 0.0


# Stable feature ordering / dimensionality (must match between prepare/predict).
FEATURE_NAMES = (
    ["ch_mean_r", "ch_mean_g", "ch_mean_b",
     "ch_std_r", "ch_std_g", "ch_std_b",
     "ch_skew_r", "ch_skew_g", "ch_skew_b",
     "brightness", "contrast", "saturation",
     "grad_mean", "grad_std", "lap_mean_abs", "lap_std",
     "fft_high_ratio"]
    + [f"fft_band_{i}" for i in range(6)]
    + ["resid_std", "resid_meanabs", "resid_kurt",
       "corr_rg", "corr_rb", "corr_gb",
       "hist_ent_r", "hist_ent_g", "hist_ent_b"]
    # Residual-spectrum fingerprint (generation artifacts), leakage-safe.
    + [f"sp_radial_{i}" for i in range(16)]
    + ["sp_outer_peakz", "sp_outer_std", "sp_logpow_kurt"]
    # Per-channel high-frequency residual stats (chroma generation artifacts).
    + ["sp_ch_outer_r", "sp_ch_outer_g", "sp_ch_outer_b",
       "sp_ch_peakz_r", "sp_ch_peakz_g", "sp_ch_peakz_b"]
)
FEATURE_DIM = len(FEATURE_NAMES)  # 57


def engineered_features(img_u8: np.ndarray) -> np.ndarray:
    """Compute the leakage-safe feature vector for one uint8 [H,W,3] image."""
    img = img_u8.astype(np.float32) / 255.0
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    feats: list[float] = []

    # Per-channel mean / std / skew.
    for ch in (r, g, b):
        feats.append(float(ch.mean()))
    for ch in (r, g, b):
        feats.append(float(ch.std()))
    for ch in (r, g, b):
        _, _, sk, _ = _moments(ch)
        feats.append(sk)

    # Overall brightness / contrast / saturation.
    feats.append(float(gray.mean()))
    feats.append(float(gray.std()))
    cmax = img.max(axis=2)
    cmin = img.min(axis=2)
    feats.append(float((cmax - cmin).mean()))

    # Edges: gradient magnitude + discrete Laplacian.
    gy, gx = np.gradient(gray)
    gmag = np.sqrt(gx * gx + gy * gy)
    feats.append(float(gmag.mean()))
    feats.append(float(gmag.std()))
    lap = (np.pad(gray, 1, mode="edge"))
    lap4 = (lap[0:-2, 1:-1] + lap[2:, 1:-1] + lap[1:-1, 0:-2]
            + lap[1:-1, 2:] - 4.0 * gray)
    feats.append(float(np.abs(lap4).mean()))
    feats.append(float(lap4.std()))

    # Frequency: 2D FFT power spectrum of mean-removed gray.
    f = np.fft.fftshift(np.fft.fft2(gray - gray.mean()))
    power = (f.real ** 2 + f.imag ** 2)
    total = power.sum() + 1e-8
    h, w = gray.shape
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = radius.max() + 1e-8
    rn = radius / rmax
    # High-frequency energy ratio (outer half of the spectrum).
    feats.append(float(power[rn > 0.5].sum() / total))
    # Radial power bands (6 equal-width rings, normalised).
    edges = np.linspace(0.0, 1.0, 7)
    for i in range(6):
        m = (rn >= edges[i]) & (rn < edges[i + 1])
        feats.append(float(power[m].sum() / total))

    # Noise residual = gray - 3x3 blur.
    resid = gray - _box_blur3(gray)
    _, rsd, _, rku = _moments(resid)
    feats.append(rsd)
    feats.append(float(np.abs(resid).mean()))
    feats.append(rku)

    # Inter-channel correlations.
    def _corr(a, b_):
        a = a.ravel(); b_ = b_.ravel()
        a = a - a.mean(); b_ = b_ - b_.mean()
        d = (np.sqrt((a * a).sum()) * np.sqrt((b_ * b_).sum())) + 1e-8
        return float((a * b_).sum() / d)
    feats.append(_corr(r, g))
    feats.append(_corr(r, b))
    feats.append(_corr(g, b))

    # Per-channel intensity histogram entropy (32 bins).
    for ch in (r, g, b):
        hist, _ = np.histogram(ch, bins=32, range=(0.0, 1.0))
        p = hist.astype(np.float64)
        p = p / (p.sum() + 1e-8)
        nz = p[p > 0]
        ent = float(-(nz * np.log2(nz)).sum())
        feats.append(ent)

    # Residual-spectrum fingerprint (16 radial bins + 3 summary stats).
    radial, peakz, osd, lk = _residual_spectrum(gray)
    feats.extend(float(v) for v in radial)
    feats.append(peakz)
    feats.append(osd)
    feats.append(lk)

    # Per-channel high-frequency residual stats (outer mean, then peakz).
    ch_outer = []
    ch_peakz = []
    for ch in (r, g, b):
        omu, pz = _channel_hf_stats(ch)
        ch_outer.append(omu)
        ch_peakz.append(pz)
    feats.extend(ch_outer)
    feats.extend(ch_peakz)

    return np.asarray(feats, dtype=np.float32)


# ----------------------------------------------------------------------------
# CNN from scratch (Appendix B exact architecture, parameterised width k)
# ----------------------------------------------------------------------------

def make_cnn(k: int = CNN_K):
    """CNN classifier head, Appendix-B structure (3->k->2k->4k) **plus BatchNorm
    and dropout**.

    The brief's Appendix B is explicitly "a starting point, not a recommended
    final solution". A from-scratch CNN without normalisation trains slowly and
    plateaus early on CPU; inserting BatchNorm after every conv accelerates and
    stabilises convergence (far better accuracy within the same epoch budget),
    and a small dropout before the classifier improves generalisation to the
    hidden holdout. The conv channel progression and AdaptiveAvgPool head are
    unchanged, so the model stays resolution-agnostic.
    """
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(3, k, 3, padding=1), nn.BatchNorm2d(k), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(k, 2 * k, 3, padding=1), nn.BatchNorm2d(2 * k), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(2 * k, 4 * k, 3, padding=1), nn.BatchNorm2d(4 * k), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(4 * k, 2),
    )


# ----------------------------------------------------------------------------
# Streaming readers
# ----------------------------------------------------------------------------

def split_dir(split: str) -> str:
    return os.path.join(DATA_DIR, split)


def list_shards(split: str) -> list[str]:
    d = split_dir(split)
    files = [f for f in os.listdir(d) if f.endswith(".parquet")]
    files.sort()
    return [os.path.join(d, f) for f in files]


def iter_split(split: str, columns: list[str]):
    """Yield pyarrow record batches across all shards of a labeled split."""
    import pyarrow.parquet as pq
    for path in list_shards(split):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=columns):
            yield batch


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Binary metrics with the AI class (==1) as positive.

    recall_ai = TP/(TP+FN); fpr_real = FP/(FP+TN) where a "false positive" is a
    real image (label 0) predicted AI (1).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    recall_ai = tp / max(1, tp + fn)
    fpr_real = fp / max(1, fp + tn)
    precision = tp / max(1, tp + fp)
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    return {
        "recall_ai": round(recall_ai, 4),
        "fpr_real": round(fpr_real, 4),
        "precision": round(precision, 4),
        "accuracy": round(accuracy, 4),
        "confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp},
    }


def calibrate_threshold(p_ai: np.ndarray, y_true: np.ndarray,
                        target_fpr: float = CALIB_TARGET_FPR) -> float:
    """Most permissive threshold on P(ai) with FPR_real <= target_fpr.

    Sweeps candidate thresholds; among those satisfying the FPR constraint picks
    the lowest (maximises recall_ai). Falls back to the threshold with minimum
    FPR if none satisfy the constraint.
    """
    p_ai = np.asarray(p_ai, dtype=np.float64)
    y_true = np.asarray(y_true).astype(int)
    cands = np.unique(np.concatenate([[0.0, 1.0], np.round(p_ai, 4)]))
    cands.sort()
    best_t = 0.5
    best_recall = -1.0
    fallback_t = 0.5
    fallback_fpr = 1.0
    for t in cands:
        pred = (p_ai >= t).astype(int)
        m = compute_metrics(y_true, pred)
        if m["fpr_real"] < fallback_fpr:
            fallback_fpr = m["fpr_real"]
            fallback_t = float(t)
        if m["fpr_real"] <= target_fpr and m["recall_ai"] > best_recall:
            best_recall = m["recall_ai"]
            best_t = float(t)
    return best_t if best_recall >= 0.0 else fallback_t
