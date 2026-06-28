#!/usr/bin/env python3
"""Task 1.2 - train two model families, calibrate, select the best, persist it.

Run:  python train.py --timeout_seconds 1800

Loads the caches written by prepare.py from ``artifacts/prepared/`` and trains:

  Family A - classical baseline on standardised leakage-safe engineered features
             (LogisticRegression + GradientBoosting, both class-balanced).
  Family B - CNN from scratch (Appendix-B architecture, weighted CrossEntropy).

Objective: maximise ``recall_ai`` subject to ``FPR_real <= 20%``. The decision
threshold on P(ai) is calibrated automatically on ``data/calibration`` (cached
as the ``calibration`` split) with a safety margin (target FPR 18%), then the
hard 20% gate is verified on ``validation``. The better family on validation is
persisted to ``artifacts/task02/``.

Timeout-aware: the CNN epoch loop is bounded by ``Deadline`` and writes the best
checkpoint after every epoch, so a kill still leaves a usable model.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import common as C
from common import (Deadline, calibrate_threshold, compute_metrics,
                    make_cnn, MAX_FPR)


# ----------------------------------------------------------------------------
# Cache loading
# ----------------------------------------------------------------------------

def load_split(split: str, mmap=True):
    """Return (images_memmap_or_None, feats, labels) for a prepared split."""
    base = os.path.join(C.PREP, split)
    img_p = base + "_images.npy"
    feat_p = base + "_feats.npy"
    lbl_p = base + "_labels.npy"
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


def load_scaler():
    z = np.load(os.path.join(C.PREP, "feature_scaler.npz"))
    return z["mean"].astype(np.float32), z["std"].astype(np.float32)


# ----------------------------------------------------------------------------
# Family A - classical baseline
# ----------------------------------------------------------------------------

def train_classical(f_tr, y_tr, mean, std):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier

    Xtr = (f_tr - mean) / std
    models = {}
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    lr.fit(Xtr, y_tr)
    models["logreg"] = lr
    try:
        gb = GradientBoostingClassifier(random_state=C.SEED)
        gb.fit(Xtr, y_tr)
        models["gboost"] = gb
    except Exception as exc:
        print(f"[train] gboost skipped: {exc}")
    # HistGradientBoosting: faster and usually stronger on the wider
    # (spectral-augmented) feature vector; class_weight balances the imbalance.
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        hgb = HistGradientBoostingClassifier(
            random_state=C.SEED, max_iter=400, learning_rate=0.06,
            max_leaf_nodes=31, l2_regularization=1.0,
            class_weight="balanced", early_stopping=True,
            validation_fraction=0.15)
        hgb.fit(Xtr, y_tr)
        models["histgb"] = hgb
    except Exception as exc:
        print(f"[train] histgb skipped: {exc}")
    return models


def classical_proba(model, f, mean, std):
    X = (f - mean) / std
    return model.predict_proba(X)[:, 1]


# ----------------------------------------------------------------------------
# Family B - CNN
# ----------------------------------------------------------------------------

def cnn_scores(model, imgs, batch=256):
    """P(ai) for an image memmap [N,128,128,3] uint8 -> float [N].

    Images are downsampled to the 64px CNN view to match training.
    """
    import torch
    model.eval()
    out = np.zeros((len(imgs),), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(imgs), batch):
            chunk = np.ascontiguousarray(imgs[i:i + batch])
            chunk = C.downsample_u8(chunk).astype(np.float32)
            chunk /= 255.0
            xb = torch.from_numpy(chunk).permute(0, 3, 1, 2).contiguous()
            logits = model(xb)
            p = torch.softmax(logits, dim=1)[:, 1]
            out[i:i + batch] = p.numpy()
    return out


def train_cnn(imgs_tr, y_tr, val_imgs, y_val, deadline: Deadline, k: int):
    import torch
    import torch.nn as nn

    model = make_cnn(k)
    # Damped class weighting: sqrt(inverse-freq). Full inverse-freq over-pushes
    # toward the "real" minority and hurts recall_ai; the post-hoc threshold
    # calibration already handles the FPR trade-off, so the loss should mainly
    # produce well-ranked probabilities.
    n0 = int((y_tr == 0).sum()); n1 = int((y_tr == 1).sum())
    w = torch.tensor([1.0 / max(1, n0), 1.0 / max(1, n1)], dtype=torch.float32)
    w = torch.sqrt(w)
    w = w / w.sum() * 2.0
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    n = len(y_tr)
    bs = 256
    rng = np.random.default_rng(C.SEED)
    flip_rng = np.random.default_rng(C.SEED + 1)
    ckpt = os.path.join(C.TASK02, "cnn_best.pt")

    best_recall = -1.0
    best_state = None
    history = []
    epoch = 0
    decayed = False
    # Reserve time to evaluate + checkpoint after the current epoch.
    while not deadline.expired():
        epoch += 1
        # Late LR decay (cheap stabiliser once most of the budget is spent).
        if not decayed and deadline.elapsed() > 0.6 * deadline.budget:
            for pg in opt.param_groups:
                pg["lr"] = 3e-4
            decayed = True
        model.train()
        idx = rng.permutation(n)
        total_loss = 0.0
        nb = 0
        for s in range(0, n, bs):
            if deadline.remaining() < 20.0:  # leave room to eval+save
                break
            bidx = np.sort(idx[s:s + bs])
            # memmap fancy-index copies only this batch into RAM, then we take
            # the cheaper 64px view for the CNN.
            chunk = np.ascontiguousarray(imgs_tr[bidx])
            chunk = C.downsample_u8(chunk).astype(np.float32)
            chunk /= 255.0
            # Seeded random horizontal flip (regularisation, reproducible).
            if flip_rng.random() < 0.5:
                chunk = chunk[:, :, ::-1, :].copy()
            xb = torch.from_numpy(chunk).permute(0, 3, 1, 2).contiguous()
            yb = torch.from_numpy(y_tr[bidx].astype(np.int64))
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            nb += 1

        # Evaluate on validation and checkpoint the best (by recall@FPR<=0.20).
        p_val = cnn_scores(model, val_imgs)
        t = calibrate_threshold(p_val, y_val)
        m = compute_metrics(y_val, (p_val >= t).astype(int))
        history.append({"epoch": epoch, "loss": round(total_loss / max(1, nb), 4),
                        "thr": round(t, 4), **m})
        print(f"[train] cnn k={k} epoch {epoch} loss={total_loss/max(1,nb):.4f} "
              f"val recall_ai={m['recall_ai']} fpr={m['fpr_real']} thr={t:.3f} "
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
    return model, history


# ----------------------------------------------------------------------------
# Evaluation helper
# ----------------------------------------------------------------------------

def eval_at_threshold(p, y, t):
    return compute_metrics(y, (np.asarray(p) >= t).astype(int))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Task 1.2 train")
    ap.add_argument("--timeout_seconds", type=float, default=1800.0)
    args = ap.parse_args()

    C.set_determinism()
    C.set_threads()
    os.makedirs(C.TASK02, exist_ok=True)
    deadline = Deadline(args.timeout_seconds)

    # Load caches.
    imgs_tr, f_tr, y_tr = load_split("train")
    imgs_cal, f_cal, y_cal = load_split("calibration")
    imgs_val, f_val, y_val = load_split("validation")
    imgs_vaug, f_vaug, y_vaug = load_split("validation_augmented")
    if y_tr is None or y_cal is None or y_val is None:
        raise SystemExit("[train] missing prepared caches; run prepare.py first")
    mean, std = load_scaler()
    print(f"[train] train={len(y_tr)} cal={len(y_cal)} val={len(y_val)} "
          f"vaug={0 if y_vaug is None else len(y_vaug)}")

    report = {"families": {}}

    # ---- Family A: classical ----
    classical = train_classical(f_tr, y_tr, mean, std)
    best_clf_name, best_clf, best_clf_thr, best_clf_val = None, None, 0.5, None
    for name, mdl in classical.items():
        p_cal = classical_proba(mdl, f_cal, mean, std)
        thr = calibrate_threshold(p_cal, y_cal)
        mval = eval_at_threshold(classical_proba(mdl, f_val, mean, std), y_val, thr)
        report["families"][f"classical_{name}"] = {
            "threshold": round(thr, 4), "validation": mval}
        print(f"[train] classical {name}: val recall_ai={mval['recall_ai']} "
              f"fpr={mval['fpr_real']} thr={thr:.3f}")
        ok = mval["fpr_real"] <= MAX_FPR
        cur = mval["recall_ai"] - (0.0 if ok else 1.0)
        prev = -2.0 if best_clf_val is None else (
            best_clf_val["recall_ai"] - (0.0 if best_clf_val["fpr_real"] <= MAX_FPR else 1.0))
        if cur > prev:
            best_clf_name, best_clf, best_clf_thr, best_clf_val = name, mdl, thr, mval

    # ---- Family B: CNN ----
    cnn_model, cnn_thr, cnn_val, cnn_hist = None, 0.5, None, []
    if imgs_tr is not None and imgs_val is not None:
        # Pick width to fit budget; smaller k if little time remains.
        k = C.CNN_K
        if deadline.remaining() < 300:
            k = 16
        elif deadline.remaining() < 700:
            k = 24
        try:
            cnn_model, cnn_hist = train_cnn(imgs_tr, y_tr, imgs_val, y_val,
                                            deadline, k)
            # Calibrate on calibration split, verify on validation.
            p_cal = cnn_scores(cnn_model, imgs_cal)
            cnn_thr = calibrate_threshold(p_cal, y_cal)
            cnn_val = eval_at_threshold(cnn_scores(cnn_model, imgs_val), y_val,
                                        cnn_thr)
            report["families"]["cnn"] = {
                "k": k, "threshold": round(cnn_thr, 4),
                "validation": cnn_val, "history": cnn_hist}
            print(f"[train] cnn: val recall_ai={cnn_val['recall_ai']} "
                  f"fpr={cnn_val['fpr_real']} thr={cnn_thr:.3f}")
        except Exception as exc:
            print(f"[train] CNN training failed: {exc}")

    # ---- Ensemble candidate: weighted average of classical + CNN P(ai) ----
    # The two families see different signals (engineered spectral/noise stats vs
    # learned spatial features), so a blend often beats either alone. We sweep
    # the blend weight on the CALIBRATION split (never validation) to avoid
    # peeking, calibrate the threshold there, then verify on validation.
    ens_thr, ens_val, ens_w = 0.5, None, 0.5
    if (best_clf is not None and cnn_model is not None
            and imgs_cal is not None and imgs_val is not None):
        try:
            pc_cal = classical_proba(best_clf, f_cal, mean, std)
            pn_cal = cnn_scores(cnn_model, imgs_cal)
            pc_val = classical_proba(best_clf, f_val, mean, std)
            pn_val = cnn_scores(cnn_model, imgs_val)
            best_w_recall = -1.0
            for w in (0.3, 0.4, 0.5, 0.6, 0.7):
                p_cal = w * pc_cal + (1.0 - w) * pn_cal
                t = calibrate_threshold(p_cal, y_cal)
                m_cal = eval_at_threshold(p_cal, y_cal, t)
                sc = m_cal["recall_ai"] - (0.0 if m_cal["fpr_real"] <= MAX_FPR else 1.0)
                if sc > best_w_recall:
                    best_w_recall, ens_w, ens_thr = sc, w, t
            p_val = ens_w * pc_val + (1.0 - ens_w) * pn_val
            ens_val = eval_at_threshold(p_val, y_val, ens_thr)
            report["families"]["ensemble"] = {
                "weight_classical": ens_w, "threshold": round(ens_thr, 4),
                "validation": ens_val,
                "members": [f"classical_{best_clf_name}", "cnn"]}
            print(f"[train] ensemble: w_clf={ens_w} "
                  f"val recall_ai={ens_val['recall_ai']} "
                  f"fpr={ens_val['fpr_real']} thr={ens_thr:.3f}")
        except Exception as exc:
            print(f"[train] ensemble failed: {exc}")

    # ---- Select winner (higher val recall_ai among FPR<=20%) ----
    def score(m):
        if m is None:
            return -2.0
        return m["recall_ai"] - (0.0 if m["fpr_real"] <= MAX_FPR else 1.0)

    cand = {"classical": (best_clf_val, best_clf_thr),
            "cnn": (cnn_val, cnn_thr),
            "ensemble": (ens_val, ens_thr)}
    winner_kind = max(cand, key=lambda kk: score(cand[kk][0]))
    meta = {"max_fpr": MAX_FPR, "feature_dim": C.FEATURE_DIM}

    def save_classical():
        import joblib
        joblib.dump(best_clf, os.path.join(C.TASK02, "classical.joblib"))
        np.savez(os.path.join(C.TASK02, "feature_scaler.npz"),
                 mean=mean, std=std)

    def save_cnn():
        import torch
        cnn_k = report["families"]["cnn"]["k"]
        torch.save({"state_dict": cnn_model.state_dict(), "k": cnn_k},
                   os.path.join(C.TASK02, "cnn_best.pt"))
        return cnn_k

    if winner_kind == "cnn":
        winner = "cnn"
        thr = cnn_thr
        val_metrics = cnn_val
        cnn_k = save_cnn()
        meta.update({"family": "cnn", "k": cnn_k, "model_file": "cnn_best.pt"})
    elif winner_kind == "ensemble":
        winner = "ensemble"
        thr = ens_thr
        val_metrics = ens_val
        save_classical()
        cnn_k = save_cnn()
        meta.update({"family": "ensemble", "k": cnn_k,
                     "classical_name": best_clf_name,
                     "weight_classical": ens_w,
                     "model_files": ["classical.joblib", "cnn_best.pt"]})
    else:
        winner = f"classical_{best_clf_name}"
        thr = best_clf_thr
        val_metrics = best_clf_val
        save_classical()
        meta.update({"family": "classical", "classical_name": best_clf_name,
                     "model_file": "classical.joblib"})

    meta["threshold"] = round(float(thr), 4)
    meta["validation"] = val_metrics

    # validation_augmented report (Task 1.3 preview; not a gate here).
    if imgs_vaug is not None and f_vaug is not None:
        if winner_kind == "cnn":
            p_vaug = cnn_scores(cnn_model, imgs_vaug)
        elif winner_kind == "ensemble":
            p_vaug = (ens_w * classical_proba(best_clf, f_vaug, mean, std)
                      + (1.0 - ens_w) * cnn_scores(cnn_model, imgs_vaug))
        else:
            p_vaug = classical_proba(best_clf, f_vaug, mean, std)
        meta["validation_augmented"] = eval_at_threshold(p_vaug, y_vaug, thr)

    with open(os.path.join(C.TASK02, "model_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    report["winner"] = winner
    report["elapsed_seconds"] = round(deadline.elapsed(), 1)
    with open(os.path.join(C.TASK02, "train_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    gate = "PASS" if val_metrics["fpr_real"] <= MAX_FPR else "FAIL"
    print(f"[train] WINNER={winner} thr={meta['threshold']} "
          f"val recall_ai={val_metrics['recall_ai']} "
          f"fpr={val_metrics['fpr_real']} gate(FPR<=20%)={gate} "
          f"({deadline.elapsed():.1f}s)")


if __name__ == "__main__":
    main()
