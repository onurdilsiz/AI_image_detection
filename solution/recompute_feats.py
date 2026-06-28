#!/usr/bin/env python3
"""Recompute cached engineered features from the cached square-resized images.

The images caches (``<split>_images.npy``, 128px) are produced by prepare.py and
do not depend on the feature definition. When ``common.engineered_features`` is
extended (e.g. new spectral fingerprint dims) we only need to recompute the
``<split>_feats.npy`` arrays and refit ``feature_scaler.npz`` on TRAIN ONLY -
no need to re-decode every image over a slow bind mount. Pure numpy, CPU-only.

Run:  python recompute_feats.py
"""

from __future__ import annotations

import os
import numpy as np

import common as C
from common import engineered_features, FEATURE_DIM, LABELED_SPLITS


def recompute_split(split: str) -> int:
    img_p = os.path.join(C.PREP, f"{split}_images.npy")
    if not os.path.exists(img_p):
        print(f"[recompute] skip {split}: no images cache")
        return 0
    imgs = np.load(img_p, mmap_mode="r")
    n = len(imgs)
    feats = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    for i in range(n):
        feats[i] = engineered_features(np.ascontiguousarray(imgs[i]))
        if (i + 1) % 5000 == 0:
            print(f"[recompute] {split}: {i + 1}/{n}")
    np.save(os.path.join(C.PREP, f"{split}_feats.npy"), feats)
    print(f"[recompute] {split}: wrote {n} x {FEATURE_DIM}")
    return n


def main():
    C.set_determinism()
    for split in LABELED_SPLITS:
        recompute_split(split)
    ftr = np.load(os.path.join(C.PREP, "train_feats.npy"))
    mean = ftr.mean(axis=0).astype(np.float32)
    std = ftr.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    np.savez(os.path.join(C.PREP, "feature_scaler.npz"), mean=mean, std=std)
    print(f"[recompute] feature_scaler refit (dim={mean.shape[0]})")


if __name__ == "__main__":
    main()
