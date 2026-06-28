#!/usr/bin/env python3
"""Recalibrate the Task 3 (augmented) operating point without retraining.

The augmented run calibrates its threshold on ``calibration_augmented``; the
calibration->validation_augmented FPR gap runs *conservative* (vaug FPR lands
~0.03 below the calibration target), so the deployed model sits at FPR ~0.14 -
well under the 20% gate - leaving recall on the table. This reloads the already
trained task03 models and recalibrates each family's threshold on
``calibration_augmented`` at a target that spends the FPR budget, then selects
the family with the best ``validation_augmented`` recall_ai subject to
FPR<=20% (the same rule train_augmented uses) and rewrites task03 model_meta.

Run:  python recalibrate_aug.py --target_fpr 0.20
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import common as C
from common import calibrate_threshold, MAX_FPR
from train import classical_proba, cnn_scores, eval_at_threshold


def load_split(split):
    base = os.path.join(C.PREP, split)
    y = np.load(base + "_labels.npy")
    f = np.load(base + "_feats.npy")
    imgs = np.load(base + "_images.npy", mmap_mode="r")
    n = min(len(y), len(f), len(imgs))
    return imgs[:n], f[:n], y[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_fpr", type=float, default=0.20)
    args = ap.parse_args()
    C.set_determinism(); C.set_threads()

    import joblib, torch
    clf = joblib.load(os.path.join(C.TASK03, "classical.joblib"))
    z = np.load(os.path.join(C.TASK03, "feature_scaler.npz"))
    mean, std = z["mean"].astype(np.float32), z["std"].astype(np.float32)
    ck = torch.load(os.path.join(C.TASK03, "cnn_best.pt"), map_location="cpu",
                    weights_only=False)
    cnn = C.make_cnn(int(ck.get("k", C.CNN_K)))
    cnn.load_state_dict(ck["state_dict"]); cnn.eval()
    cnn_k = int(ck.get("k", C.CNN_K))
    with open(os.path.join(C.TASK03, "model_meta.json")) as fh:
        clf_name = json.load(fh).get("classical_name", "histgb")

    imgs_cal, f_cal, y_cal = load_split("calibration_augmented")   # calibrate
    imgs_g, f_g, y_g = load_split("validation_augmented")          # gate/select
    imgs_v, f_v, y_v = load_split("validation")                    # clean report
    t = args.target_fpr

    pc_cal, pc_g, pc_v = (classical_proba(clf, f_cal, mean, std),
                          classical_proba(clf, f_g, mean, std),
                          classical_proba(clf, f_v, mean, std))
    pn_cal, pn_g, pn_v = cnn_scores(cnn, imgs_cal), cnn_scores(cnn, imgs_g), cnn_scores(cnn, imgs_v)

    cands = {}
    thr_c = calibrate_threshold(pc_cal, y_cal, t)
    cands["classical"] = (eval_at_threshold(pc_g, y_g, thr_c), thr_c, 1.0)
    thr_n = calibrate_threshold(pn_cal, y_cal, t)
    cands["cnn"] = (eval_at_threshold(pn_g, y_g, thr_n), thr_n, 0.0)
    best_w, best_thr, best_score = 0.5, 0.5, -1.0
    for w in (0.2, 0.3, 0.4, 0.5, 0.6):
        p_cal = w * pc_cal + (1.0 - w) * pn_cal
        th = calibrate_threshold(p_cal, y_cal, t)
        mc = eval_at_threshold(p_cal, y_cal, th)
        sc = mc["recall_ai"] - (0.0 if mc["fpr_real"] <= MAX_FPR else 1.0)
        if sc > best_score:
            best_score, best_w, best_thr = sc, w, th
    p_g_ens = best_w * pc_g + (1.0 - best_w) * pn_g
    cands["ensemble"] = (eval_at_threshold(p_g_ens, y_g, best_thr), best_thr, best_w)

    for name, (m, th, w) in cands.items():
        print(f"[recal_aug] {name}: vaug recall_ai={m['recall_ai']} fpr={m['fpr_real']} "
              f"thr={th:.3f} w_clf={w}")

    def score(m):
        return m["recall_ai"] - (0.0 if m["fpr_real"] <= MAX_FPR else 1.0)
    winner = max(cands, key=lambda k: score(cands[k][0]))
    m, thr, w = cands[winner]
    gate = "PASS" if m["fpr_real"] <= MAX_FPR else "FAIL"
    print(f"[recal_aug] WINNER={winner} vaug recall_ai={m['recall_ai']} "
          f"fpr={m['fpr_real']} gate={gate}")

    meta = {"max_fpr": MAX_FPR, "feature_dim": C.FEATURE_DIM,
            "continued_from": "artifacts/task02/cnn_best.pt",
            "calibrated_target_fpr": t, "threshold": round(float(thr), 4),
            "validation_augmented": m}
    if winner == "classical":
        meta.update({"family": "classical", "classical_name": clf_name,
                     "model_file": "classical.joblib"}); p_v = pc_v
    elif winner == "cnn":
        meta.update({"family": "cnn", "k": cnn_k, "model_file": "cnn_best.pt"}); p_v = pn_v
    else:
        meta.update({"family": "ensemble", "k": cnn_k, "classical_name": clf_name,
                     "weight_classical": w,
                     "model_files": ["classical.joblib", "cnn_best.pt"]})
        p_v = w * pc_v + (1.0 - w) * pn_v
    meta["validation"] = eval_at_threshold(p_v, y_v, thr)
    print(f"[recal_aug] clean val recall_ai={meta['validation']['recall_ai']} "
          f"fpr={meta['validation']['fpr_real']}")

    with open(os.path.join(C.TASK03, "model_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[recal_aug] task03 model_meta.json updated (family={meta['family']})")


if __name__ == "__main__":
    main()
