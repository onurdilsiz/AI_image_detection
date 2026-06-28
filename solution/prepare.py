#!/usr/bin/env python3
"""Task 1.2 - prepare cached, model-ready arrays from the cleaned splits.

Run:  python prepare.py --timeout_seconds 600

Reads the read-only parquet splits under ``data/`` and the cleaning manifest
``artifacts/clean/clean_manifest.parquet`` (produced by clean.py), and writes
model-ready caches under ``artifacts/prepared/``:

  <split>_images.npy   uint8  [N,128,128,3]   (square-resized; shortcut removed)
  <split>_feats.npy    float32[N,D]           (leakage-safe engineered features)
  <split>_labels.npy   int8   [N]
  feature_scaler.npz   mean/std fit on TRAIN ONLY (no val/calib leakage)
  prepared_summary.json

Design
------
* TRAIN rows are selected by *replaying the manifest* (keep only the cleaned
  rows per shard). Other labeled splits keep every row (label from source_class).
* ``data/predict/`` is intentionally NOT prepared - it can change at grading
  time; predict.py decodes it inline instead.
* Images are streamed and written into a memmap so peak RAM stays bounded
  (train cache ~ 29k x 128 x 128 x 3 = ~1.4 GB on disk, never all in RAM).
* Deadline-aware: train (largest) is processed first; whatever completes is
  persisted with a ``complete`` flag in prepared_summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

import common as C
from common import (Deadline, decode_resize, engineered_features, to_label,
                    FEATURE_DIM, RESIZE)


# ----------------------------------------------------------------------------
# Manifest replay (train only)
# ----------------------------------------------------------------------------

def load_train_keep():
    """Return dict[shard_basename] -> dict[row_index] -> label from manifest."""
    import pyarrow.parquet as pq
    if not os.path.exists(C.CLEAN_MANIFEST):
        raise FileNotFoundError(
            f"missing {C.CLEAN_MANIFEST}; run clean.py first")
    t = pq.read_table(C.CLEAN_MANIFEST,
                      columns=["shard_file", "row_index", "label"])
    shard = t.column("shard_file").to_pylist()
    ridx = t.column("row_index").to_pylist()
    lbl = t.column("label").to_pylist()
    keep: dict[str, dict[int, int]] = defaultdict(dict)
    for s, r, l in zip(shard, ridx, lbl):
        keep[s][int(r)] = int(l)
    return keep


# ----------------------------------------------------------------------------
# Per-split preparation
# ----------------------------------------------------------------------------

def _open_memmap(path: str, n: int):
    return np.lib.format.open_memmap(
        path, mode="w+", dtype=np.uint8, shape=(n, RESIZE, RESIZE, 3))


def prepare_train(keep, deadline: Deadline):
    """Stream train shards, decode kept rows into the image memmap + features."""
    # Upper bound on kept rows (we shrink at the end to the number decoded).
    n_keep = sum(len(v) for v in keep.values())
    img_path = os.path.join(C.PREP, "train_images.npy")
    images = _open_memmap(img_path, n_keep)
    feats = np.zeros((n_keep, FEATURE_DIM), dtype=np.float32)
    labels = np.zeros((n_keep,), dtype=np.int8)

    written = 0
    stopped = False
    for path in C.list_shards("train"):
        if stopped:
            break
        base = os.path.basename(path)
        keep_rows = keep.get(base, {})
        if not keep_rows:
            continue
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        row = 0
        for batch in pf.iter_batches(batch_size=C.BATCH_SIZE,
                                     columns=["image"]):
            if deadline.expired():
                stopped = True
                break
            imgs = batch.column("image")
            for j in range(batch.num_rows):
                ridx = row
                row += 1
                if ridx not in keep_rows:
                    continue
                arr = decode_resize(imgs[j].as_py())
                if arr is None:
                    continue
                images[written] = arr
                feats[written] = engineered_features(arr)
                labels[written] = keep_rows[ridx]
                written += 1
    # Truncate caches to the number actually decoded (stay label-aligned).
    images.flush()
    del images
    src = np.lib.format.open_memmap(
        img_path, mode="r", dtype=np.uint8, shape=(n_keep, RESIZE, RESIZE, 3))
    tmp = img_path + ".tmp"
    out = np.lib.format.open_memmap(
        tmp, mode="w+", dtype=np.uint8, shape=(written, RESIZE, RESIZE, 3))
    out[:] = src[:written]
    out.flush()
    del out, src
    os.replace(tmp, img_path)
    np.save(os.path.join(C.PREP, "train_feats.npy"), feats[:written])
    np.save(os.path.join(C.PREP, "train_labels.npy"), labels[:written])
    return written, (not stopped)


def prepare_labeled_split(split: str, deadline: Deadline):
    """Decode every row of a (small) labeled split into image/feat/label npy."""
    # Count rows first (cheap metadata scan) to size the memmap.
    import pyarrow.parquet as pq
    n_total = 0
    for path in C.list_shards(split):
        n_total += pq.ParquetFile(path).metadata.num_rows

    img_path = os.path.join(C.PREP, f"{split}_images.npy")
    images = _open_memmap(img_path, n_total)
    feats = np.zeros((n_total, FEATURE_DIM), dtype=np.float32)
    labels = np.zeros((n_total,), dtype=np.int8)

    written = 0
    stopped = False
    for batch in C.iter_split(split, ["image", "source_class"]):
        if deadline.expired():
            stopped = True
            break
        imgs = batch.column("image")
        scls = batch.column("source_class").to_pylist()
        for j in range(batch.num_rows):
            arr = decode_resize(imgs[j].as_py())
            if arr is None:
                continue
            images[written] = arr
            feats[written] = engineered_features(arr)
            labels[written] = to_label(scls[j])
            written += 1

    images.flush()
    tmp = img_path + ".tmp"
    src = np.lib.format.open_memmap(
        img_path, mode="r", dtype=np.uint8, shape=(n_total, RESIZE, RESIZE, 3))
    out = np.lib.format.open_memmap(
        tmp, mode="w+", dtype=np.uint8, shape=(written, RESIZE, RESIZE, 3))
    out[:] = src[:written]
    out.flush()
    del out, src, images
    os.replace(tmp, img_path)
    np.save(os.path.join(C.PREP, f"{split}_feats.npy"), feats[:written])
    np.save(os.path.join(C.PREP, f"{split}_labels.npy"), labels[:written])
    return written, (not stopped)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Task 1.2 prepare")
    ap.add_argument("--timeout_seconds", type=float, default=600.0)
    args = ap.parse_args()

    C.set_determinism()
    os.makedirs(C.PREP, exist_ok=True)
    deadline = Deadline(args.timeout_seconds)

    summary = {"splits": {}, "feature_dim": FEATURE_DIM,
               "resize": RESIZE, "complete": False}

    # 1) Train first (largest, via manifest replay).
    keep = load_train_keep()
    n_tr, done_tr = prepare_train(keep, deadline)
    y = np.load(os.path.join(C.PREP, "train_labels.npy"))
    summary["splits"]["train"] = {
        "rows": int(n_tr), "complete": bool(done_tr),
        "n_real": int((y == 0).sum()), "n_ai": int((y == 1).sum())}
    print(f"[prepare] train: {n_tr} rows complete={done_tr} "
          f"real={int((y==0).sum())} ai={int((y==1).sum())} "
          f"({deadline.elapsed():.1f}s)")

    # 2) Feature scaler fit on TRAIN ONLY.
    if n_tr > 0:
        ftr = np.load(os.path.join(C.PREP, "train_feats.npy"))
        mean = ftr.mean(axis=0).astype(np.float32)
        std = ftr.std(axis=0).astype(np.float32)
        std[std < 1e-6] = 1.0
        np.savez(os.path.join(C.PREP, "feature_scaler.npz"),
                 mean=mean, std=std)
        print(f"[prepare] feature_scaler saved (dim={mean.shape[0]})")

    # 3) Remaining labeled splits.
    for split in [s for s in C.LABELED_SPLITS if s != "train"]:
        if deadline.expired():
            print(f"[prepare] skipping {split}: out of time")
            summary["splits"][split] = {"rows": 0, "complete": False}
            continue
        n, done = prepare_labeled_split(split, deadline)
        ys = np.load(os.path.join(C.PREP, f"{split}_labels.npy"))
        summary["splits"][split] = {
            "rows": int(n), "complete": bool(done),
            "n_real": int((ys == 0).sum()), "n_ai": int((ys == 1).sum())}
        print(f"[prepare] {split}: {n} rows complete={done} "
              f"real={int((ys==0).sum())} ai={int((ys==1).sum())} "
              f"({deadline.elapsed():.1f}s)")

    summary["complete"] = all(
        v.get("complete") for v in summary["splits"].values())
    summary["elapsed_seconds"] = round(deadline.elapsed(), 1)
    with open(os.path.join(C.PREP, "prepared_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[prepare] DONE complete={summary['complete']} "
          f"({deadline.elapsed():.1f}s)")


if __name__ == "__main__":
    main()
