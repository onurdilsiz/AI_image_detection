#!/usr/bin/env python3
"""Task 1.3 - run the robust (augmented) model on data/predict.

Run:  python predict_augmented.py --timeout_seconds 600

Loads the winning robust model + calibrated threshold from
``artifacts/task03/`` and streams ``data/predict/*.parquet``, decoding each
image inline. Writes ``artifacts/task03/predictions.csv`` in the same format as
Task 2 (header ``row_id,predicted_label``; ``1`` == ai_generated iff
``P(ai) >= threshold``; unreadable images fall back to label 0).
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

import common as C
from common import Deadline, decode_resize, engineered_features


def _load_cnn():
    import torch
    ck = torch.load(os.path.join(C.TASK03, "cnn_best.pt"),
                    map_location="cpu", weights_only=False)
    model = C.make_cnn(int(ck.get("k", C.CNN_K)))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


def _load_classical():
    import joblib
    model = joblib.load(os.path.join(C.TASK03, "classical.joblib"))
    z = np.load(os.path.join(C.TASK03, "feature_scaler.npz"))
    scaler = (z["mean"].astype(np.float32), z["std"].astype(np.float32))
    return model, scaler


def load_winner():
    """Return (family, cnn_model_or_None, classical_or_None, scaler_or_None, thr, w)."""
    with open(os.path.join(C.TASK03, "model_meta.json")) as fh:
        meta = json.load(fh)
    family = meta["family"]
    thr = float(meta["threshold"])
    w = float(meta.get("weight_classical", 0.5))
    if family == "cnn":
        return ("cnn", _load_cnn(), None, None, thr, w)
    if family == "ensemble":
        clf, scaler = _load_classical()
        return ("ensemble", _load_cnn(), clf, scaler, thr, w)
    clf, scaler = _load_classical()
    return ("classical", None, clf, scaler, thr, w)


def _cnn_p(model, arr):
    import torch
    x = C.downsample_u8(arr).astype(np.float32) / 255.0
    xb = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).contiguous()
    with torch.no_grad():
        return float(torch.softmax(model(xb), dim=1)[0, 1])


def _classical_p(model, scaler, arr):
    mean, std = scaler
    f = (engineered_features(arr) - mean) / std
    return float(model.predict_proba(f.reshape(1, -1))[0, 1])


def main():
    ap = argparse.ArgumentParser(description="Task 1.3 predict_augmented")
    ap.add_argument("--timeout_seconds", type=float, default=600.0)
    args = ap.parse_args()

    C.set_determinism()
    C.set_threads()
    os.makedirs(C.TASK03, exist_ok=True)
    deadline = Deadline(args.timeout_seconds)

    family, cnn_model, clf, scaler, thr, w = load_winner()
    print(f"[predict_aug] family={family} threshold={thr:.4f} w_clf={w:.2f}")

    rows = []
    import pyarrow.parquet as pq
    for path in C.list_shards("predict"):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=C.BATCH_SIZE,
                                     columns=["row_id", "image"]):
            rids = batch.column("row_id").to_pylist()
            imgs = batch.column("image")
            for j in range(batch.num_rows):
                rid = int(rids[j])
                arr = decode_resize(imgs[j].as_py())
                if arr is None:
                    rows.append((rid, 0))
                    continue
                if family == "cnn":
                    p = _cnn_p(cnn_model, arr)
                elif family == "ensemble":
                    p = (w * _classical_p(clf, scaler, arr)
                         + (1.0 - w) * _cnn_p(cnn_model, arr))
                else:
                    p = _classical_p(clf, scaler, arr)
                rows.append((rid, 1 if p >= thr else 0))

    rows.sort(key=lambda r: r[0])
    out_p = os.path.join(C.TASK03, "predictions.csv")
    with open(out_p, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["row_id", "predicted_label"])
        wr.writerows(rows)
    n_ai = sum(1 for _, l in rows if l == 1)
    print(f"[predict_aug] wrote {len(rows)} rows -> {out_p} "
          f"(ai={n_ai} real={len(rows)-n_ai}) ({deadline.elapsed():.1f}s)")


if __name__ == "__main__":
    main()
