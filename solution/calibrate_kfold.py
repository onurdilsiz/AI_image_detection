#!/usr/bin/env python3
"""K-fold calibration stability check.

Computes thresholds via leave-one-fold-out calibration on the prepared
`calibration` split and reports distribution of thresholds + held-out metrics.

Run from `solution/`:
  python calibrate_kfold.py --k 5
"""
from __future__ import annotations

import json
import os
import argparse
import numpy as np

import common as C


def load_calibration():
    base = os.path.join(C.ART, "prepared")
    feats = np.load(os.path.join(base, "calibration_feats.npy"))
    labels = np.load(os.path.join(base, "calibration_labels.npy"))
    imgs = np.load(os.path.join(base, "calibration_images.npy"))
    return feats, labels.astype(int), imgs


def compute_probs(family, cnn_model, clf, scaler, w, feats, imgs):
    # returns np.array of p(ai) for each sample
    n = len(feats)
    p = np.zeros(n, dtype=float)
    if family in ("classical", "ensemble") and clf is not None:
        mean, std = scaler
        X = (feats - mean) / std
        p_clf = clf.predict_proba(X)[:, 1]
    else:
        p_clf = None
    if family in ("cnn", "ensemble") and cnn_model is not None:
        # compute cnn probs per image
        import predict as P
        p_cnn = np.zeros(n, dtype=float)
        for i in range(n):
            p_cnn[i] = P._cnn_p(cnn_model, imgs[i])
    else:
        p_cnn = None

    if family == "classical":
        return p_clf
    if family == "cnn":
        return p_cnn
    # ensemble
    return w * p_clf + (1.0 - w) * p_cnn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()

    C.set_determinism(args.seed)
    C.set_threads()

    feats, labels, imgs = load_calibration()
    import predict as P
    family, cnn_model, clf, scaler, thr, w = P.load_winner()
    print(f"[kfold] family={family} base_threshold={thr:.4f} w={w:.2f}")

    n = len(labels)
    rng = np.random.default_rng(args.seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, args.k)

    results = []
    for i in range(len(folds)):
        test_idx = folds[i]
        train_idx = np.concatenate([f for j, f in enumerate(folds) if j != i])

        p_all = compute_probs(family, cnn_model, clf, scaler, w, feats, imgs)
        p_train = p_all[train_idx]
        y_train = labels[train_idx]
        p_test = p_all[test_idx]
        y_test = labels[test_idx]

        t = C.calibrate_threshold(p_train, y_train, target_fpr=C.CALIB_TARGET_FPR)
        preds = (p_test >= t).astype(int)
        m = C.compute_metrics(y_test, preds)
        results.append({"fold": i, "threshold": float(t), "metrics": m})
        print(f"[kfold] fold={i} thr={t:.4f} recall={m['recall_ai']:.4f} fpr={m['fpr_real']:.4f}")

    out = os.path.join(C.TASK02, "kfold_calibration.json")
    os.makedirs(C.TASK02, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[kfold] wrote {out}")


if __name__ == "__main__":
    main()
