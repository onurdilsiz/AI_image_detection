#!/usr/bin/env python3
"""Robustness curves evaluation script.

Applies a variety of perturbations at multiple severities to the validation set
and records recall / FPR under the deployed threshold.

Run from `solution/`:
  python robustness_eval.py
Results are saved to `artifacts/task02/robustness_curves.json`.
"""
from __future__ import annotations

import json
import os
import argparse
import numpy as np

import common as C


def load_validation():
    base = os.path.join(C.ART, "prepared")
    imgs = np.load(os.path.join(base, "validation_images.npy"))
    labels = np.load(os.path.join(base, "validation_labels.npy"))
    return imgs, labels.astype(int)


def compute_probs_for_imgs(family, cnn_model, clf, scaler, w, imgs):
    import predict as P
    n = len(imgs)
    p_cnn = None
    p_clf = None
    if family in ("cnn", "ensemble") and cnn_model is not None:
        p_cnn = np.zeros(n, dtype=float)
        for i in range(n):
            p_cnn[i] = P._cnn_p(cnn_model, imgs[i])
    if family in ("classical", "ensemble") and clf is not None:
        # need feats for classical
        feats = np.zeros((n, C.FEATURE_DIM), dtype=np.float32)
        for i in range(n):
            feats[i] = C.engineered_features(imgs[i])
        mean, std = scaler
        X = (feats - mean) / std
        p_clf = clf.predict_proba(X)[:, 1]
    if family == "cnn":
        return p_cnn
    if family == "classical":
        return p_clf
    return w * p_clf + (1.0 - w) * p_cnn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--levels", type=int, default=5,
                    help="number of severity levels (including zero)")
    args = ap.parse_args()

    C.set_determinism(args.seed)
    C.set_threads()

    imgs, labels = load_validation()
    import predict as P
    family, cnn_model, clf, scaler, thr, w = P.load_winner()
    print(f"[robust] family={family} threshold={thr:.4f} w={w:.2f}")

    modes = ["jpeg", "blur", "downscale", "noise", "color_jitter",
             "crop", "rotate", "saltpepper", "combined"]
    levels = np.linspace(0.0, 1.0, args.levels)
    rng = np.random.default_rng(args.seed)

    curves = {}
    for mode in modes:
        rows = []
        for sev in levels:
            # build augmented set deterministically
            aug_imgs = np.zeros_like(imgs)
            for i in range(len(imgs)):
                # use a per-sample rng to keep reproducible
                r = np.random.default_rng(args.seed + i)
                aug_imgs[i] = C.augment_extended_u8(imgs[i], r, mode=mode, severity=float(sev))

            p = compute_probs_for_imgs(family, cnn_model, clf, scaler, w, aug_imgs)
            preds = (p >= thr).astype(int)
            m = C.compute_metrics(labels, preds)
            rows.append({"severity": float(sev), "metrics": m})
            print(f"[robust] mode={mode} sev={sev:.2f} recall={m['recall_ai']:.4f} fpr={m['fpr_real']:.4f}")
        curves[mode] = rows

    out = os.path.join(C.TASK02, "robustness_curves.json")
    os.makedirs(C.TASK02, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(curves, fh, indent=2)
    print(f"[robust] wrote {out}")


if __name__ == "__main__":
    main()
