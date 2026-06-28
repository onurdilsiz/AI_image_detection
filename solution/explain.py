#!/usr/bin/env python3
"""Task 1.4 - explainability for the final detector.

Run:  python explain.py --task 02 --n_samples 8 --timeout_seconds 600

Produces visual + quantitative explanations for the selected model and writes
them to ``artifacts/task0X/explain/``. Covers the four directions in the brief:

  1. **Classical feature importance** - which of the 57 leakage-safe features
     drive the tree/linear model, grouped into *spectral/high-frequency* vs
     *content/colour* so we can see how much the decision leans on the (fragile)
     generation fingerprint.
  2. **CNN saliency** - gradient of P(ai) w.r.t. input pixels for sample images.
  3. **Occlusion analysis** - slide an occluding patch and measure the change in
     P(ai) to localise the evidence.
  4. **FP / FN analysis + real-vs-AI attention** - confusion breakdown, per-group
     feature means, example montages, and mean saliency magnitude for real vs AI.

Explanations are reported *critically*: the accompanying ``summary.json`` and the
printed discussion flag where the evidence looks like a genuine content cue vs a
spectral shortcut that augmentation (Task 1.3) can erase.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C
from common import Deadline, compute_metrics, FEATURE_NAMES


# Indices of the spectral / high-frequency feature block (Task 1.2 fingerprint).
def _spectral_mask():
    mask = np.zeros(len(FEATURE_NAMES), dtype=bool)
    for i, nm in enumerate(FEATURE_NAMES):
        if nm.startswith(("fft_", "sp_", "resid_")):
            mask[i] = True
    return mask


def _task_dir(task: str) -> str:
    return C.TASK03 if str(task) == "03" else C.TASK02


def _load_meta(task_dir: str) -> dict:
    with open(os.path.join(task_dir, "model_meta.json")) as fh:
        return json.load(fh)


def _load_prepared(split: str):
    base = os.path.join(C.PREP, split)
    imgs = np.load(base + "_images.npy", mmap_mode="r")
    feats = np.load(base + "_feats.npy")
    y = np.load(base + "_labels.npy")
    n = min(len(imgs), len(feats), len(y))
    return imgs[:n], feats[:n], y[:n]


def _load_classical(task_dir):
    import joblib
    p = os.path.join(task_dir, "classical.joblib")
    if not os.path.exists(p):
        return None, None
    model = joblib.load(p)
    z = np.load(os.path.join(task_dir, "feature_scaler.npz"))
    return model, (z["mean"].astype(np.float32), z["std"].astype(np.float32))


def _load_cnn(task_dir):
    p = os.path.join(task_dir, "cnn_best.pt")
    if not os.path.exists(p):
        return None
    import torch
    ck = torch.load(p, map_location="cpu", weights_only=False)
    model = C.make_cnn(int(ck.get("k", C.CNN_K)))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


def _classical_p(model, scaler, feats):
    mean, std = scaler
    return model.predict_proba((feats - mean) / std)[:, 1]


def _cnn_p_batch(model, imgs, batch=128):
    import torch
    out = np.zeros(len(imgs), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(imgs), batch):
            chunk = C.downsample_u8(np.ascontiguousarray(imgs[i:i + batch]))
            chunk = chunk.astype(np.float32) / 255.0
            xb = torch.from_numpy(chunk).permute(0, 3, 1, 2).contiguous()
            out[i:i + batch] = torch.softmax(model(xb), 1)[:, 1].numpy()
    return out


# ----------------------------------------------------------------------------
# 1. Classical feature importance
# ----------------------------------------------------------------------------

def explain_feature_importance(model, X_scaled, y, out_dir, summary):
    names = list(FEATURE_NAMES)
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=np.float64)
        kind = "impurity/gain importance"
    elif hasattr(model, "coef_"):
        imp = np.abs(np.asarray(model.coef_, dtype=np.float64)).ravel()
        kind = "abs(logistic coefficient)"
    else:
        # Model-agnostic fallback (e.g. HistGradientBoosting): permutation
        # importance by drop in ROC-AUC when each feature is shuffled.
        try:
            from sklearn.inspection import permutation_importance
            r = permutation_importance(model, X_scaled, y, scoring="roc_auc",
                                       n_repeats=3, random_state=C.SEED)
            imp = np.clip(np.asarray(r.importances_mean, dtype=np.float64), 0, None)
            kind = "permutation importance (ROC-AUC drop)"
        except Exception as exc:
            print(f"[explain] feature importance skipped: {exc}")
            return
    imp = imp / (imp.sum() + 1e-12)
    spec = _spectral_mask()
    order = np.argsort(imp)[::-1]
    top = order[:20]

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#d62728" if spec[i] else "#1f77b4" for i in top]
    ax.barh([names[i] for i in top][::-1], imp[top][::-1],
            color=colors[::-1])
    ax.set_xlabel(f"normalised importance ({kind})")
    ax.set_title("Top-20 classical feature importances\n"
                 "red = spectral/high-freq fingerprint, blue = content/colour")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=110)
    plt.close(fig)

    spectral_share = float(imp[spec].sum())
    summary["feature_importance"] = {
        "kind": kind,
        "spectral_share": round(spectral_share, 4),
        "content_share": round(1.0 - spectral_share, 4),
        "top5": [{"name": names[i], "importance": round(float(imp[i]), 4)}
                 for i in order[:5]],
    }
    print(f"[explain] feature importance: spectral block carries "
          f"{spectral_share:.1%} of the model's importance")


# ----------------------------------------------------------------------------
# 2/4. CNN saliency + real-vs-AI attention
# ----------------------------------------------------------------------------

def _saliency_map(model, img128):
    """|d P(ai) / d input| aggregated over channels, at the 64px CNN view."""
    import torch
    x = C.downsample_u8(img128).astype(np.float32) / 255.0
    xb = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).contiguous()
    xb.requires_grad_(True)
    p = torch.softmax(model(xb), 1)[0, 1]
    model.zero_grad()
    p.backward()
    sal = xb.grad.detach().abs().squeeze(0).max(0).values.numpy()
    return float(p.detach()), sal


def explain_saliency(model, imgs, y, p_final, thr, out_dir, summary, n=8):
    # Pick representative samples: TP, FP, TN, FN.
    pred = (p_final >= thr).astype(int)
    groups = {
        "TP": np.where((y == 1) & (pred == 1))[0],
        "FN": np.where((y == 1) & (pred == 0))[0],
        "TN": np.where((y == 0) & (pred == 0))[0],
        "FP": np.where((y == 0) & (pred == 1))[0],
    }
    per = max(1, n // 4)
    picks = []
    for g, idx in groups.items():
        for j in idx[:per]:
            picks.append((g, int(j)))
    if not picks:
        return

    rows = len(picks)
    fig, axes = plt.subplots(rows, 2, figsize=(5, 2.4 * rows))
    if rows == 1:
        axes = axes[None, :]
    for r, (g, j) in enumerate(picks):
        img = np.ascontiguousarray(imgs[j])
        p, sal = _saliency_map(model, img)
        axes[r, 0].imshow(img)
        axes[r, 0].set_title(f"{g} y={int(y[j])} P(ai)={p:.2f}", fontsize=8)
        axes[r, 0].axis("off")
        axes[r, 1].imshow(sal, cmap="inferno")
        axes[r, 1].set_title("saliency |dP/dx|", fontsize=8)
        axes[r, 1].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "saliency_montage.png"), dpi=110)
    plt.close(fig)

    # Mean saliency magnitude for real vs AI (attention comparison).
    real_idx = np.where(y == 0)[0][:24]
    ai_idx = np.where(y == 1)[0][:24]
    real_m = float(np.mean([_saliency_map(model, np.ascontiguousarray(imgs[j]))[1].mean()
                            for j in real_idx])) if len(real_idx) else 0.0
    ai_m = float(np.mean([_saliency_map(model, np.ascontiguousarray(imgs[j]))[1].mean()
                          for j in ai_idx])) if len(ai_idx) else 0.0
    summary["real_vs_ai_saliency"] = {
        "mean_saliency_real": round(real_m, 6),
        "mean_saliency_ai": round(ai_m, 6)}
    print(f"[explain] mean saliency real={real_m:.5f} ai={ai_m:.5f}")


# ----------------------------------------------------------------------------
# 3. Occlusion analysis
# ----------------------------------------------------------------------------

def explain_occlusion(model, imgs, y, p_final, thr, out_dir, patch=16, stride=16):
    import torch
    pred = (p_final >= thr).astype(int)
    cand = np.where((y == 1) & (pred == 1))[0]
    if len(cand) == 0:
        cand = np.where(y == 1)[0]
    if len(cand) == 0:
        return
    j = int(cand[0])
    img = np.ascontiguousarray(imgs[j])
    base_x = C.downsample_u8(img).astype(np.float32) / 255.0
    with torch.no_grad():
        base_p = float(torch.softmax(model(torch.from_numpy(base_x)
                       .permute(2, 0, 1).unsqueeze(0)), 1)[0, 1])
    H = img.shape[0]
    grid = range(0, H, stride)
    heat = np.zeros((len(list(grid)), len(list(range(0, H, stride)))), np.float32)
    gray = int(0.5 * 255)
    with torch.no_grad():
        for a, yy in enumerate(range(0, H, stride)):
            for b, xx in enumerate(range(0, H, stride)):
                occ = img.copy()
                occ[yy:yy + patch, xx:xx + patch] = gray
                xv = C.downsample_u8(occ).astype(np.float32) / 255.0
                p = float(torch.softmax(model(torch.from_numpy(xv)
                          .permute(2, 0, 1).unsqueeze(0)), 1)[0, 1])
                heat[a, b] = base_p - p  # positive = patch was evidence for AI
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.4))
    ax[0].imshow(img); ax[0].set_title(f"AI image P(ai)={base_p:.2f}", fontsize=9)
    ax[0].axis("off")
    im = ax[1].imshow(heat, cmap="coolwarm")
    ax[1].set_title("occlusion: drop in P(ai)", fontsize=9); ax[1].axis("off")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "occlusion.png"), dpi=110)
    plt.close(fig)
    print(f"[explain] occlusion map written (base P(ai)={base_p:.2f})")


# ----------------------------------------------------------------------------
# 4. FP / FN analysis
# ----------------------------------------------------------------------------

def explain_fp_fn(imgs, feats, y, p_final, thr, out_dir, summary):
    pred = (p_final >= thr).astype(int)
    m = compute_metrics(y, pred)
    summary["confusion"] = m["confusion"]
    summary["validation_metrics"] = {
        "recall_ai": m["recall_ai"], "fpr_real": m["fpr_real"],
        "precision": m["precision"], "accuracy": m["accuracy"]}

    spec = _spectral_mask()
    groups = {
        "TP": (y == 1) & (pred == 1), "FN": (y == 1) & (pred == 0),
        "TN": (y == 0) & (pred == 0), "FP": (y == 0) & (pred == 1)}
    grp_stats = {}
    for g, msk in groups.items():
        if msk.sum() == 0:
            continue
        grp_stats[g] = {
            "n": int(msk.sum()),
            "mean_p_ai": round(float(p_final[msk].mean()), 4),
            "mean_spectral_feat": round(float(feats[msk][:, spec].mean()), 4)}
    summary["group_stats"] = grp_stats

    # Example montage of the worst FP and FN (highest-confidence mistakes).
    fp_idx = np.where(groups["FP"])[0]
    fn_idx = np.where(groups["FN"])[0]
    fp_idx = fp_idx[np.argsort(p_final[fp_idx])[::-1]][:4] if len(fp_idx) else fp_idx
    fn_idx = fn_idx[np.argsort(p_final[fn_idx])][:4] if len(fn_idx) else fn_idx
    picks = [("FP", j) for j in fp_idx] + [("FN", j) for j in fn_idx]
    if picks:
        cols = len(picks)
        fig, axes = plt.subplots(1, cols, figsize=(2.2 * cols, 2.6))
        if cols == 1:
            axes = [axes]
        for ax, (g, j) in zip(axes, picks):
            ax.imshow(np.ascontiguousarray(imgs[j]))
            ax.set_title(f"{g} P(ai)={p_final[j]:.2f}", fontsize=8)
            ax.axis("off")
        fig.suptitle("Highest-confidence errors (FP = real->AI, FN = AI->real)",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fp_fn_examples.png"), dpi=110)
        plt.close(fig)
    print(f"[explain] FP/FN analysis: {m['confusion']}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Task 1.4 explainability")
    ap.add_argument("--task", default="02", choices=["02", "03"])
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--timeout_seconds", type=float, default=600.0)
    args = ap.parse_args()

    C.set_determinism()
    C.set_threads()
    Deadline(args.timeout_seconds)
    task_dir = _task_dir(args.task)
    out_dir = os.path.join(task_dir, "explain")
    os.makedirs(out_dir, exist_ok=True)

    meta = _load_meta(task_dir)
    family = meta["family"]
    thr = float(meta["threshold"])
    w = float(meta.get("weight_classical", 0.5))
    print(f"[explain] task={args.task} family={family} thr={thr:.4f}")

    imgs, feats, y = _load_prepared(args.split)
    clf, scaler = _load_classical(task_dir)
    try:
        cnn = _load_cnn(task_dir)
    except Exception as exc:
        print(f"[explain] CNN unavailable ({exc}); classical explanations only")
        cnn = None

    # Final P(ai) of the deployed model on the split.
    p_clf = _classical_p(clf, scaler, feats) if clf is not None else None
    p_cnn = _cnn_p_batch(cnn, imgs) if cnn is not None else None
    if family == "ensemble" and p_cnn is not None and p_clf is not None:
        p_final = w * p_clf + (1.0 - w) * p_cnn
    elif family == "cnn" and p_cnn is not None:
        p_final = p_cnn
    elif p_clf is not None:
        if family != "classical":
            print(f"[explain] note: CNN missing; explaining the classical "
                  f"sub-model of the {family} winner")
        p_final = p_clf
    else:
        p_final = p_cnn
    if p_final is None:
        raise SystemExit("[explain] no usable model probabilities")

    summary = {"task": args.task, "split": args.split, "family": family,
               "threshold": thr}

    if clf is not None:
        Xs = (feats - scaler[0]) / scaler[1]
        explain_feature_importance(clf, Xs, y, out_dir, summary)
    if cnn is not None:
        try:
            explain_saliency(cnn, imgs, y, p_final, thr, out_dir, summary,
                             n=args.n_samples)
            explain_occlusion(cnn, imgs, y, p_final, thr, out_dir)
        except Exception as exc:
            print(f"[explain] CNN explanations failed: {exc}")
    explain_fp_fn(imgs, feats, y, p_final, thr, out_dir, summary)

    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[explain] wrote explanations -> {out_dir}")


if __name__ == "__main__":
    main()
