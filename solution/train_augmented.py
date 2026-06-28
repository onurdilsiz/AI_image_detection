#!/usr/bin/env python3
"""Task 1.3 - augmentation / feature engineering for robustness.

Run:  python train_augmented.py --timeout_seconds 1800

Goal: a detector that stays accurate when images are *scaled, compressed,
blurred or noised*. We continue from the Task 2 starting point and harden it:

  * **Classical** (the Task 2 winner family): retrain LogReg / GradientBoosting /
    HistGradientBoosting on the leakage-safe features of BOTH the clean training
    images AND a randomly augmented copy of each (``common.augment_u8``). The
    standardiser is refit on this augmented distribution. This is cheap on CPU
    and directly teaches the model the perturbation-invariant feature regions.
  * **CNN** (continuing from ``artifacts/task02/cnn_best.pt``): fine-tuned on the
    64px view with on-the-fly augmentation, bounded by the time budget.

Operating point is **recalibrated on ``calibration_augmented``** (the augmented
calibration split) under the same target FPR, and the hard 20% gate is verified
on ``validation_augmented``. We also report clean ``validation`` for the
Task 2 vs Task 3 comparison the brief asks for. The best family on
``validation_augmented`` (recall_ai subject to FPR<=20%) is persisted to
``artifacts/task03/``.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import common as C
from common import (Deadline, calibrate_threshold, compute_metrics, MAX_FPR,
                    engineered_features, make_cnn)
from train import (train_classical, classical_proba, cnn_scores,
                   eval_at_threshold)


def load_split(split: str, mmap=True):
    base = os.path.join(C.PREP, split)
    img_p, feat_p, lbl_p = base + "_images.npy", base + "_feats.npy", base + "_labels.npy"
    if not (os.path.exists(feat_p) and os.path.exists(lbl_p)):
        return None, None, None
    y = np.load(lbl_p)
    f = np.load(feat_p)
    imgs = None
    if os.path.exists(img_p):
        imgs = np.load(img_p, mmap_mode="r" if mmap else None)
        n = min(len(imgs), len(y))
        imgs, f, y = imgs[:n], f[:n], y[:n]
    return imgs, f, y


# ----------------------------------------------------------------------------
# Augmented training features (clean + one random augmentation per image)
# ----------------------------------------------------------------------------

def augmented_train_feats(imgs, f_clean, y, deadline: Deadline,
                          reserve: float = 240.0):
    """Return (X, y2): clean features stacked with augmented-image features.

    ``f_clean`` is the cached clean-image feature matrix; we only have to
    compute features for the augmented copies. Deadline-aware: if time runs
    short we use however many augmented rows we managed (always >=0), so the
    classical fit still has the full clean set plus a partial augmented set.
    """
    rng = np.random.default_rng(C.SEED + 7)
    n = len(y)
    f_aug = np.zeros((n, f_clean.shape[1]), dtype=np.float32)
    done = 0
    for i in range(n):
        if deadline.remaining() < reserve:
            break
        a = C.augment_u8(np.ascontiguousarray(imgs[i]), rng)
        f_aug[i] = engineered_features(a)
        done += 1
        if (i + 1) % 5000 == 0:
            print(f"[train_aug] augmented feats {i + 1}/{n} "
                  f"({deadline.elapsed():.0f}s)")
    if done == 0:
        return f_clean.copy(), np.asarray(y).copy()
    X = np.concatenate([f_clean, f_aug[:done]], axis=0)
    y2 = np.concatenate([np.asarray(y), np.asarray(y)[:done]], axis=0)
    print(f"[train_aug] training set: {len(f_clean)} clean + {done} augmented "
          f"= {len(X)} rows")
    return X, y2


# ----------------------------------------------------------------------------
# CNN fine-tune from the Task 2 checkpoint, with on-the-fly augmentation
# ----------------------------------------------------------------------------

def finetune_cnn(imgs_tr, y_tr, val_imgs, y_val, deadline: Deadline):
    """Continue training the Task 2 CNN on augmented 64px batches.

    Loads ``artifacts/task02/cnn_best.pt`` if present (the "continue from the
    Task 2 starting point" requirement); otherwise starts a fresh CNN. Best
    checkpoint by recall@FPR<=20% on validation_augmented is saved to task03.
    """
    import torch
    import torch.nn as nn

    k = C.CNN_K
    ckpt_in = os.path.join(C.TASK02, "cnn_best.pt")
    model = make_cnn(k)
    if os.path.exists(ckpt_in):
        try:
            ck = torch.load(ckpt_in, map_location="cpu", weights_only=False)
            k = int(ck.get("k", k))
            model = make_cnn(k)
            model.load_state_dict(ck["state_dict"])
            print(f"[train_aug] CNN continues from task02 checkpoint (k={k})")
        except Exception as exc:
            print(f"[train_aug] CNN checkpoint load failed ({exc}); fresh start")

    n0 = int((y_tr == 0).sum()); n1 = int((y_tr == 1).sum())
    w = torch.sqrt(torch.tensor([1.0 / max(1, n0), 1.0 / max(1, n1)]))
    w = w / w.sum() * 2.0
    crit = nn.CrossEntropyLoss(weight=w.float())
    # Gentle fine-tune of the strong clean checkpoint: a small LR + weight decay
    # adapts it to the augmented distribution without memorising it (high LR
    # overfits the augmented train set and drops validation_augmented recall).
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    n = len(y_tr)
    bs = 256
    rng = np.random.default_rng(C.SEED)
    aug_rng = np.random.default_rng(C.SEED + 11)
    ckpt = os.path.join(C.TASK03, "cnn_best.pt")
    best_recall, best_state, history, epoch = -1.0, None, [], 0
    decayed = False

    while not deadline.expired():
        epoch += 1
        if not decayed and deadline.elapsed() > 0.6 * deadline.budget:
            for pg in opt.param_groups:
                pg["lr"] = 6e-5
            decayed = True
        model.train()
        idx = rng.permutation(n)
        total_loss, nb = 0.0, 0
        for s in range(0, n, bs):
            if deadline.remaining() < 25.0:
                break
            bidx = np.sort(idx[s:s + bs])
            chunk = np.ascontiguousarray(imgs_tr[bidx])
            # Augment ~50% of the batch on the fly: a balanced mix of clean and
            # perturbed views generalises to validation_augmented better than an
            # all-augmented batch (which the model memorises).
            for j in range(len(bidx)):
                if aug_rng.random() < 0.5:
                    chunk[j] = C.augment_u8(chunk[j], aug_rng)
            chunk = C.downsample_u8(chunk).astype(np.float32) / 255.0
            if aug_rng.random() < 0.5:
                chunk = chunk[:, :, ::-1, :].copy()
            xb = torch.from_numpy(chunk).permute(0, 3, 1, 2).contiguous()
            yb = torch.from_numpy(y_tr[bidx].astype(np.int64))
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()); nb += 1

        p_val = cnn_scores(model, val_imgs)
        t = calibrate_threshold(p_val, y_val)
        m = compute_metrics(y_val, (p_val >= t).astype(int))
        history.append({"epoch": epoch, "loss": round(total_loss / max(1, nb), 4),
                        "thr": round(t, 4), **m})
        print(f"[train_aug] cnn ft epoch {epoch} loss={total_loss/max(1,nb):.4f} "
              f"vaug recall_ai={m['recall_ai']} fpr={m['fpr_real']} "
              f"({deadline.elapsed():.0f}s)")
        score = m["recall_ai"] if m["fpr_real"] <= MAX_FPR else m["recall_ai"] - 1.0
        if score > best_recall:
            best_recall = score
            best_state = {kk: v.clone() for kk, v in model.state_dict().items()}
            torch.save({"state_dict": best_state, "k": k}, ckpt)
        if deadline.remaining() < 30.0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, k, history


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Task 1.3 train_augmented")
    ap.add_argument("--timeout_seconds", type=float, default=1800.0)
    args = ap.parse_args()

    C.set_determinism()
    C.set_threads()
    os.makedirs(C.TASK03, exist_ok=True)
    deadline = Deadline(args.timeout_seconds)

    imgs_tr, f_tr, y_tr = load_split("train")
    _, f_cal_a, y_cal_a = load_split("calibration_augmented")
    imgs_cal_a, _, _ = load_split("calibration_augmented")
    imgs_vaug, f_vaug, y_vaug = load_split("validation_augmented")
    _, f_val, y_val = load_split("validation")           # clean (comparison)
    imgs_val, _, _ = load_split("validation")
    if y_tr is None or y_cal_a is None or y_vaug is None:
        raise SystemExit("[train_aug] missing prepared caches; run prepare.py")
    print(f"[train_aug] train={len(y_tr)} cal_aug={len(y_cal_a)} "
          f"vaug={len(y_vaug)} val={0 if y_val is None else len(y_val)}")

    report = {"families": {}}

    # ---- Robust classical on clean + augmented features ----
    X_aug, y_aug = augmented_train_feats(imgs_tr, f_tr, y_tr, deadline)
    mean = X_aug.mean(axis=0).astype(np.float32)
    std = X_aug.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    classical = train_classical(X_aug, y_aug, mean, std)

    # On augmented data the calibration->validation FPR gap runs *conservative*
    # (vaug FPR lands well below the calibration target), so we calibrate at the
    # full 20% budget to avoid leaving recall on the table; the hard gate is
    # still verified on validation_augmented below.
    aug_target = C.MAX_FPR
    best_clf_name, best_clf, best_clf_thr, best_clf_va = None, None, 0.5, None
    for name, mdl in classical.items():
        thr = calibrate_threshold(classical_proba(mdl, f_cal_a, mean, std), y_cal_a, aug_target)
        m_va = eval_at_threshold(classical_proba(mdl, f_vaug, mean, std), y_vaug, thr)
        m_v = (eval_at_threshold(classical_proba(mdl, f_val, mean, std), y_val, thr)
               if y_val is not None else None)
        report["families"][f"classical_{name}"] = {
            "threshold": round(thr, 4), "validation_augmented": m_va,
            "validation": m_v}
        print(f"[train_aug] classical {name}: vaug recall_ai={m_va['recall_ai']} "
              f"fpr={m_va['fpr_real']} thr={thr:.3f}")
        ok = m_va["fpr_real"] <= MAX_FPR
        cur = m_va["recall_ai"] - (0.0 if ok else 1.0)
        prev = -2.0 if best_clf_va is None else (
            best_clf_va["recall_ai"] - (0.0 if best_clf_va["fpr_real"] <= MAX_FPR else 1.0))
        if cur > prev:
            best_clf_name, best_clf, best_clf_thr, best_clf_va = name, mdl, thr, m_va

    # ---- CNN fine-tune from the Task 2 checkpoint (augmented) ----
    cnn_model, cnn_k, cnn_thr, cnn_va, cnn_hist = None, C.CNN_K, 0.5, None, []
    if imgs_tr is not None and imgs_vaug is not None and deadline.remaining() > 120:
        try:
            cnn_model, cnn_k, cnn_hist = finetune_cnn(
                imgs_tr, y_tr, imgs_vaug, y_vaug, deadline)
            cnn_thr = calibrate_threshold(cnn_scores(cnn_model, imgs_cal_a), y_cal_a, aug_target)
            cnn_va = eval_at_threshold(cnn_scores(cnn_model, imgs_vaug), y_vaug, cnn_thr)
            report["families"]["cnn"] = {
                "k": cnn_k, "threshold": round(cnn_thr, 4),
                "validation_augmented": cnn_va, "history": cnn_hist}
            print(f"[train_aug] cnn: vaug recall_ai={cnn_va['recall_ai']} "
                  f"fpr={cnn_va['fpr_real']} thr={cnn_thr:.3f}")
        except Exception as exc:
            print(f"[train_aug] CNN fine-tune failed: {exc}")

    # ---- Weighted ensemble, swept on calibration_augmented ----
    ens_thr, ens_va, ens_w = 0.5, None, 0.5
    if (best_clf is not None and cnn_model is not None
            and imgs_cal_a is not None and imgs_vaug is not None):
        try:
            pc_cal = classical_proba(best_clf, f_cal_a, mean, std)
            pn_cal = cnn_scores(cnn_model, imgs_cal_a)
            pc_va = classical_proba(best_clf, f_vaug, mean, std)
            pn_va = cnn_scores(cnn_model, imgs_vaug)
            best_w = -1.0
            for wgt in (0.3, 0.4, 0.5, 0.6, 0.7):
                p_cal = wgt * pc_cal + (1.0 - wgt) * pn_cal
                t = calibrate_threshold(p_cal, y_cal_a, aug_target)
                mc = eval_at_threshold(p_cal, y_cal_a, t)
                sc = mc["recall_ai"] - (0.0 if mc["fpr_real"] <= MAX_FPR else 1.0)
                if sc > best_w:
                    best_w, ens_w, ens_thr = sc, wgt, t
            p_va = ens_w * pc_va + (1.0 - ens_w) * pn_va
            ens_va = eval_at_threshold(p_va, y_vaug, ens_thr)
            report["families"]["ensemble"] = {
                "weight_classical": ens_w, "threshold": round(ens_thr, 4),
                "validation_augmented": ens_va,
                "members": [f"classical_{best_clf_name}", "cnn"]}
            print(f"[train_aug] ensemble: w_clf={ens_w} "
                  f"vaug recall_ai={ens_va['recall_ai']} fpr={ens_va['fpr_real']}")
        except Exception as exc:
            print(f"[train_aug] ensemble failed: {exc}")

    # ---- Select winner by recall_ai on validation_augmented under FPR<=20% ----
    def score(m):
        if m is None:
            return -2.0
        return m["recall_ai"] - (0.0 if m["fpr_real"] <= MAX_FPR else 1.0)

    cand = {"classical": (best_clf_va, best_clf_thr),
            "cnn": (cnn_va, cnn_thr),
            "ensemble": (ens_va, ens_thr)}
    winner_kind = max(cand, key=lambda kk: score(cand[kk][0]))
    meta = {"max_fpr": MAX_FPR, "feature_dim": C.FEATURE_DIM,
            "continued_from": "artifacts/task02/cnn_best.pt"}

    def save_classical():
        import joblib
        joblib.dump(best_clf, os.path.join(C.TASK03, "classical.joblib"))
        np.savez(os.path.join(C.TASK03, "feature_scaler.npz"), mean=mean, std=std)

    def save_cnn():
        import torch
        torch.save({"state_dict": cnn_model.state_dict(), "k": cnn_k},
                   os.path.join(C.TASK03, "cnn_best.pt"))

    if winner_kind == "cnn":
        winner, thr, va = "cnn", cnn_thr, cnn_va
        save_cnn()
        meta.update({"family": "cnn", "k": cnn_k, "model_file": "cnn_best.pt"})
    elif winner_kind == "ensemble":
        winner, thr, va = "ensemble", ens_thr, ens_va
        save_classical(); save_cnn()
        meta.update({"family": "ensemble", "k": cnn_k,
                     "classical_name": best_clf_name, "weight_classical": ens_w,
                     "model_files": ["classical.joblib", "cnn_best.pt"]})
    else:
        winner, thr, va = f"classical_{best_clf_name}", best_clf_thr, best_clf_va
        save_classical()
        meta.update({"family": "classical", "classical_name": best_clf_name,
                     "model_file": "classical.joblib"})

    meta["threshold"] = round(float(thr), 4)
    meta["validation_augmented"] = va

    # Clean-validation metrics for the winner (Task 2 vs Task 3 comparison).
    if y_val is not None:
        if winner_kind == "cnn":
            p_v = cnn_scores(cnn_model, imgs_val)
        elif winner_kind == "ensemble":
            p_v = (ens_w * classical_proba(best_clf, f_val, mean, std)
                   + (1.0 - ens_w) * cnn_scores(cnn_model, imgs_val))
        else:
            p_v = classical_proba(best_clf, f_val, mean, std)
        meta["validation"] = eval_at_threshold(p_v, y_val, thr)

    with open(os.path.join(C.TASK03, "model_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    report["winner"] = winner
    report["elapsed_seconds"] = round(deadline.elapsed(), 1)
    with open(os.path.join(C.TASK03, "train_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    gate = "PASS" if va["fpr_real"] <= MAX_FPR else "FAIL"
    vclean = meta.get("validation", {})
    print(f"[train_aug] WINNER={winner} thr={meta['threshold']} "
          f"vaug recall_ai={va['recall_ai']} fpr={va['fpr_real']} "
          f"gate(FPR<=20%)={gate} | clean val recall_ai="
          f"{vclean.get('recall_ai')} fpr={vclean.get('fpr_real')} "
          f"({deadline.elapsed():.1f}s)")


if __name__ == "__main__":
    main()
