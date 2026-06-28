#!/usr/bin/env python3
"""Task 1.1 - Dataset exploration & deterministic cleaning.

Run:  python clean.py --timeout_seconds 600

Reads the read-only training shards under ``data/train/`` and produces, under
``artifacts/`` (the only writable location):

  artifacts/task01/exploration_summary.json   - class / size / aspect / byte stats
  artifacts/task01/*.png                       - visual findings report
  artifacts/clean/clean_manifest.parquet       - the kept rows (deduped/cleaned)

Design notes
------------
* The data is ~866 MB across 7 shards, so everything is streamed shard-by-shard
  (never fully loaded). Outputs are written best-effort: the process may be
  killed at ``--timeout_seconds``.
* Cleaning is *deterministic* and *content preserving* (this is cleaning, not
  augmentation). We drop only: exact byte-duplicates, undecodable/corrupt
  images, and degenerate images (tiny min-side or pathological aspect ratio).
* We deliberately do NOT drop on the "shortcut" features (squareness, pixel
  size, file bytes) even though AI vs real are trivially separable by them -
  removing those rows would distort the label distribution. The pipeline
  instead records a fixed-square downstream resize target that *neutralises*
  the shortcut so the model must learn content. The shortcut is the headline
  finding the report calls out; here we only quantify and document it.
* The deliverable is a manifest (list of kept rows), not a materialised copy of
  the images: re-emitting cleaned images would duplicate ~866 MB and blow the
  20 MB submission limit. ``prepare.py`` replays this manifest against the
  read-only ``data/``. This satisfies "a script that regenerates the cleaned
  dataset" while keeping artifacts tiny.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import time
from collections import defaultdict

import numpy as np

# Headless rendering - must be set before pyplot import.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from PIL import Image  # noqa: E402

# ----------------------------------------------------------------------------
# Constants / paths
# ----------------------------------------------------------------------------

SEED = 0
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
ART = os.path.join(HERE, "artifacts")
TASK01 = os.path.join(ART, "task01")
CLEAN = os.path.join(ART, "clean")

# Streaming batch size (rows per pyarrow batch).
BATCH_SIZE = 512
# Cap analysis copy so per-pixel stats stay cheap regardless of source size.
ANALYSIS_MAX = 256
# Downstream preprocessing target that neutralises the size/aspect/byte shortcut.
RESIZE_TARGET = 128

# Deterministic drop thresholds (justified by the size distribution, not arbitrary).
MIN_SIDE_DROP = 32      # images whose shorter side is < 32 px are unusable
MAX_ASPECT_DROP = 5.0   # pathological strips (>5:1) are almost certainly junk

# Label mapping: real == source_class 0, everything else (AI source classes
# 1..5) is merged into the positive "ai_generated" class == 1.
def to_label(source_class: int) -> int:
    return 0 if int(source_class) == 0 else 1


# ----------------------------------------------------------------------------
# Deadline helper
# ----------------------------------------------------------------------------

class Deadline:
    """Monotonic wall-clock budget with a safety factor.

    The grader kills the process at ``timeout_seconds``; we stop the optional
    (expensive) work a little early so we always get to write outputs.
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
# IO / decode / metadata helpers
# ----------------------------------------------------------------------------

def list_train_shards() -> list[str]:
    """Sorted absolute paths of train shards (deterministic iteration order)."""
    files = [f for f in os.listdir(TRAIN_DIR) if f.endswith(".parquet")]
    files.sort()
    return [os.path.join(TRAIN_DIR, f) for f in files]


def iter_batches(path: str, columns: list[str]):
    """Yield ``pyarrow`` record batches from one shard, bounding RAM."""
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=columns):
        yield batch


def decode_rgb(raw: bytes) -> Image.Image | None:
    """Decode raw image bytes to a PIL RGB image, or ``None`` if corrupt."""
    try:
        im = Image.open(io.BytesIO(raw))
        im = im.convert("RGB")
        im.load()
        return im
    except Exception:
        return None


def image_meta(im: Image.Image, n_bytes: int) -> dict:
    """Size/shape metadata plus cheap pixel stats on a downscaled copy."""
    w, h = im.size
    min_side = min(w, h)
    max_side = max(w, h)
    aspect = max_side / max(1, min_side)
    megapixels = (w * h) / 1e6

    # Downscale for per-pixel stats so cost is bounded by ANALYSIS_MAX, not the
    # (possibly large) source resolution.
    if max_side > ANALYSIS_MAX:
        scale = ANALYSIS_MAX / max_side
        small = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    else:
        small = im
    arr = np.asarray(small, dtype=np.float32) / 255.0  # HxWx3 in [0,1]

    brightness = float(arr.mean())
    contrast = float(arr.std())
    # Saturation = (max-min) over channels, mean across pixels.
    cmax = arr.max(axis=2)
    cmin = arr.min(axis=2)
    saturation = float((cmax - cmin).mean())

    return {
        "w": int(w),
        "h": int(h),
        "min_side": int(min_side),
        "aspect": float(aspect),
        "megapixels": float(megapixels),
        "n_bytes": int(n_bytes),
        "square": bool(w == h),
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
    }


# ----------------------------------------------------------------------------
# Pass 1 - labels + byte-dedup (cheap, always completes)
# ----------------------------------------------------------------------------

def pass1_labels_and_dedup(shards: list[str]):
    """Stream all shards; collect labels and mark exact byte-duplicates.

    Returns
    -------
    labels    : dict[(shard_idx, row_idx)] -> label (0/1)
    dup_drop  : set[(shard_idx, row_idx)] of byte-duplicate rows to drop
    n_total   : int
    class_counts : dict label -> count (over all rows, pre-dedup)
    order     : list of (shard_idx, row_idx) in deterministic stream order
    """
    labels: dict[tuple[int, int], int] = {}
    dup_drop: set[tuple[int, int]] = set()
    class_counts: dict[int, int] = defaultdict(int)
    order: list[tuple[int, int]] = []
    seen: set[str] = set()
    n_total = 0

    for si, path in enumerate(shards):
        row = 0
        for batch in iter_batches(path, ["image", "source_class"]):
            imgs = batch.column("image")
            scls = batch.column("source_class").to_pylist()
            for j in range(batch.num_rows):
                key = (si, row)
                lbl = to_label(scls[j])
                labels[key] = lbl
                class_counts[lbl] += 1
                order.append(key)
                raw = imgs[j].as_py()
                digest = hashlib.sha1(raw).hexdigest()
                if digest in seen:
                    dup_drop.add(key)
                else:
                    seen.add(digest)
                n_total += 1
                row += 1

    return labels, dup_drop, n_total, dict(class_counts), order


# ----------------------------------------------------------------------------
# Pass 2 - metadata + deterministic cleaning (deadline-bounded)
# ----------------------------------------------------------------------------

def pass2_metadata_and_clean(shards, labels, dup_drop, deadline: Deadline):
    """Decode each non-duplicate row, compute metadata, apply drop rules.

    Rows not reached before the deadline are kept but flagged ``examined=False``
    (only their byte-duplicates were already removed in pass 1).
    """
    meta: dict[tuple[int, int], dict] = {}
    drop_reason: dict[tuple[int, int], str] = {}
    # A few thumbnails per class for the montage figure.
    montage: dict[int, list] = {0: [], 1: []}
    MONTAGE_PER_CLASS = 8

    n_examined = 0
    stopped_early = False

    for si, path in enumerate(shards):
        if stopped_early:
            break
        row = 0
        for batch in iter_batches(path, ["image"]):
            if deadline.expired():
                stopped_early = True
                break
            imgs = batch.column("image")
            for j in range(batch.num_rows):
                key = (si, row)
                row_idx = row
                row += 1
                if key in dup_drop:
                    continue  # already dropped as duplicate
                if deadline.expired():
                    stopped_early = True
                    break
                raw = imgs[j].as_py()
                im = decode_rgb(raw)
                if im is None:
                    drop_reason[key] = "corrupt"
                    n_examined += 1
                    continue
                m = image_meta(im, len(raw))
                meta[key] = m
                # Deterministic degeneracy drops.
                if m["min_side"] < MIN_SIDE_DROP:
                    drop_reason[key] = "min_side"
                elif m["aspect"] > MAX_ASPECT_DROP:
                    drop_reason[key] = "aspect"
                else:
                    lbl = labels[key]
                    if len(montage[lbl]) < MONTAGE_PER_CLASS:
                        thumb = im.copy()
                        thumb.thumbnail((96, 96))
                        montage[lbl].append(np.asarray(thumb))
                n_examined += 1
            if stopped_early:
                break

    return meta, drop_reason, montage, n_examined, stopped_early


# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------

def _pct(values):
    if not values:
        return {}
    a = np.asarray(values, dtype=np.float64)
    return {
        "min": float(a.min()),
        "p25": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "p75": float(np.percentile(a, 75)),
        "max": float(a.max()),
    }


def build_summary(n_total, class_counts, dup_drop, drop_reason, labels, meta,
                  n_examined, stopped_early, timeout_seconds, elapsed):
    # Per-class metric collections (over examined, non-dropped rows).
    per_class = {0: defaultdict(list), 1: defaultdict(list)}
    n_square = {0: 0, 1: 0}
    n_kept_class = {0: 0, 1: 0}
    for key, m in meta.items():
        if key in drop_reason:
            continue
        lbl = labels[key]
        per_class[lbl]["w"].append(m["w"])
        per_class[lbl]["h"].append(m["h"])
        per_class[lbl]["min_side"].append(m["min_side"])
        per_class[lbl]["aspect"].append(m["aspect"])
        per_class[lbl]["megapixels"].append(m["megapixels"])
        per_class[lbl]["n_bytes"].append(m["n_bytes"])
        per_class[lbl]["brightness"].append(m["brightness"])
        per_class[lbl]["contrast"].append(m["contrast"])
        per_class[lbl]["saturation"].append(m["saturation"])
        n_square[lbl] += int(m["square"])
        n_kept_class[lbl] += 1

    drop_counts = defaultdict(int)
    for r in drop_reason.values():
        drop_counts[r] += 1

    n_dup = len(dup_drop)
    n_corrupt = drop_counts.get("corrupt", 0)
    n_degen = drop_counts.get("min_side", 0) + drop_counts.get("aspect", 0)
    n_dropped = n_dup + n_corrupt + n_degen
    n_kept = n_total - n_dropped

    class_dist = {}
    for lbl, cnt in sorted(class_counts.items()):
        class_dist[str(lbl)] = {
            "name": "real" if lbl == 0 else "ai_generated",
            "count": int(cnt),
            "pct": round(100.0 * cnt / max(1, n_total), 2),
        }

    def class_block(lbl):
        d = per_class[lbl]
        kept = n_kept_class[lbl]
        return {
            "n_examined_kept": kept,
            "pct_square": round(100.0 * n_square[lbl] / max(1, kept), 2),
            "width": _pct(d["w"]),
            "height": _pct(d["h"]),
            "min_side": _pct(d["min_side"]),
            "aspect": _pct(d["aspect"]),
            "megapixels": _pct(d["megapixels"]),
            "n_bytes": _pct(d["n_bytes"]),
            "brightness": _pct(d["brightness"]),
            "contrast": _pct(d["contrast"]),
            "saturation": _pct(d["saturation"]),
        }

    # Quantify the shortcut: how separable are the classes on raw metadata?
    def med(lbl, k):
        v = per_class[lbl][k]
        return float(np.median(v)) if v else None

    shortcut = {
        "description": (
            "AI-generated images are uniformly square and smaller (pixels and "
            "file bytes); real images are larger and non-square. Raw size / "
            "aspect / file-byte metadata trivially separates the classes - a "
            "shortcut a classifier could exploit instead of learning content."
        ),
        "real_pct_square": round(100.0 * n_square[0] / max(1, n_kept_class[0]), 2),
        "ai_pct_square": round(100.0 * n_square[1] / max(1, n_kept_class[1]), 2),
        "real_median_min_side": med(0, "min_side"),
        "ai_median_min_side": med(1, "min_side"),
        "real_median_aspect": med(0, "aspect"),
        "ai_median_aspect": med(1, "aspect"),
        "real_median_kb": (round(med(0, "n_bytes") / 1024.0, 1)
                           if med(0, "n_bytes") else None),
        "ai_median_kb": (round(med(1, "n_bytes") / 1024.0, 1)
                         if med(1, "n_bytes") else None),
        "mitigation": (
            f"Downstream preprocessing resizes every image to a fixed "
            f"{RESIZE_TARGET}x{RESIZE_TARGET} RGB square, which removes the "
            "size/aspect/byte signal and forces the model to learn content. "
            "We therefore do NOT drop rows on these features during cleaning "
            "(that would distort the label distribution); we only document them."
        ),
    }

    return {
        "n_total_rows": int(n_total),
        "n_kept": int(n_kept),
        "n_examined": int(n_examined),
        "examined_all": (not stopped_early),
        "timeout_seconds": float(timeout_seconds),
        "elapsed_seconds": round(float(elapsed), 1),
        "stopped_early_due_to_timeout": bool(stopped_early),
        "class_distribution": class_dist,
        "drops": {
            "duplicates": int(n_dup),
            "corrupt": int(n_corrupt),
            "degenerate_min_side": int(drop_counts.get("min_side", 0)),
            "degenerate_aspect": int(drop_counts.get("aspect", 0)),
            "total_dropped": int(n_dropped),
            "rules": {
                "min_side_lt": MIN_SIDE_DROP,
                "aspect_gt": MAX_ASPECT_DROP,
                "dedup": "exact SHA1 of raw image bytes, keep first occurrence",
            },
        },
        "per_class_stats": {
            "0_real": class_block(0),
            "1_ai_generated": class_block(1),
        },
        "shortcut_findings": shortcut,
        "downstream_resize_target": [RESIZE_TARGET, RESIZE_TARGET],
        "notes": (
            "Cleaning is deterministic and content-preserving. Rows not examined "
            "before the deadline are kept (examined=False in the manifest) with "
            "only their byte-duplicates removed; a generous --timeout_seconds "
            "(~600s) examines the full set."
        ),
    }


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

COL = {0: "#2c7fb8", 1: "#d95f0e"}  # real, ai


def make_figures(summary, labels, drop_reason, meta, montage):
    # Gather kept metadata per class for plotting.
    pc = {0: defaultdict(list), 1: defaultdict(list)}
    for key, m in meta.items():
        if key in drop_reason:
            continue
        lbl = labels[key]
        pc[lbl]["w"].append(m["w"])
        pc[lbl]["h"].append(m["h"])
        pc[lbl]["aspect"].append(m["aspect"])
        pc[lbl]["kb"].append(m["n_bytes"] / 1024.0)

    # (a) class balance
    cd = summary["class_distribution"]
    fig, ax = plt.subplots(figsize=(4, 3))
    names = [cd[k]["name"] for k in sorted(cd)]
    counts = [cd[k]["count"] for k in sorted(cd)]
    cols = [COL[int(k)] for k in sorted(cd)]
    ax.bar(names, counts, color=cols)
    ax.set_title("Class distribution")
    ax.set_ylabel("count")
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(TASK01, "class_balance.png"), dpi=110)
    plt.close(fig)

    # (b) width vs height scatter
    fig, ax = plt.subplots(figsize=(4.5, 4))
    for lbl in (0, 1):
        if pc[lbl]["w"]:
            ax.scatter(pc[lbl]["w"], pc[lbl]["h"], s=4, alpha=0.3,
                       color=COL[lbl],
                       label="real" if lbl == 0 else "ai_generated")
    ax.set_xlabel("width (px)")
    ax.set_ylabel("height (px)")
    ax.set_title("Image size by class")
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(os.path.join(TASK01, "size_scatter.png"), dpi=110)
    plt.close(fig)

    # (c) aspect histogram
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for lbl in (0, 1):
        if pc[lbl]["aspect"]:
            ax.hist(pc[lbl]["aspect"], bins=40, alpha=0.5, color=COL[lbl],
                    label="real" if lbl == 0 else "ai_generated")
    ax.set_xlabel("aspect ratio (max/min side)")
    ax.set_ylabel("count")
    ax.set_title("Aspect ratio by class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(TASK01, "aspect_hist.png"), dpi=110)
    plt.close(fig)

    # (d) file-bytes histogram
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for lbl in (0, 1):
        if pc[lbl]["kb"]:
            ax.hist(pc[lbl]["kb"], bins=40, alpha=0.5, color=COL[lbl],
                    label="real" if lbl == 0 else "ai_generated")
    ax.set_xlabel("file size (KB)")
    ax.set_ylabel("count")
    ax.set_title("Encoded file size by class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(TASK01, "bytes_hist.png"), dpi=110)
    plt.close(fig)

    # (e) montage of example thumbnails
    n_per = max(len(montage[0]), len(montage[1]))
    if n_per > 0:
        fig, axes = plt.subplots(2, n_per, figsize=(1.3 * n_per, 3))
        if n_per == 1:
            axes = axes.reshape(2, 1)
        for r, lbl in enumerate((0, 1)):
            for c in range(n_per):
                ax = axes[r][c]
                ax.axis("off")
                if c < len(montage[lbl]):
                    ax.imshow(montage[lbl][c])
            axes[r][0].set_ylabel("real" if lbl == 0 else "ai",
                                  rotation=0, ha="right", va="center",
                                  fontsize=9)
        fig.suptitle("Example images (top: real, bottom: ai_generated)",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(TASK01, "montage.png"), dpi=110)
        plt.close(fig)


# ----------------------------------------------------------------------------
# Manifest
# ----------------------------------------------------------------------------

def write_manifest(shards, labels, dup_drop, drop_reason, meta, order):
    """Write the kept rows to artifacts/clean/clean_manifest.parquet."""
    shard_names = [os.path.basename(p) for p in shards]
    cols = {
        "shard_file": [],
        "row_index": [],
        "label": [],
        "width": [],
        "height": [],
        "n_bytes": [],
        "examined": [],
    }
    for key in order:
        if key in dup_drop or key in drop_reason:
            continue
        si, row = key
        m = meta.get(key)
        cols["shard_file"].append(shard_names[si])
        cols["row_index"].append(int(row))
        cols["label"].append(int(labels[key]))
        cols["width"].append(int(m["w"]) if m else -1)
        cols["height"].append(int(m["h"]) if m else -1)
        cols["n_bytes"].append(int(m["n_bytes"]) if m else -1)
        cols["examined"].append(bool(m is not None))

    table = pa.table({
        "shard_file": pa.array(cols["shard_file"], pa.string()),
        "row_index": pa.array(cols["row_index"], pa.int32()),
        "label": pa.array(cols["label"], pa.int8()),
        "width": pa.array(cols["width"], pa.int32()),
        "height": pa.array(cols["height"], pa.int32()),
        "n_bytes": pa.array(cols["n_bytes"], pa.int32()),
        "examined": pa.array(cols["examined"], pa.bool_()),
    })
    pq.write_table(table, os.path.join(CLEAN, "clean_manifest.parquet"))
    return table.num_rows


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Task 1.1 exploration + cleaning")
    ap.add_argument("--timeout_seconds", type=float, default=600.0)
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    os.makedirs(TASK01, exist_ok=True)
    os.makedirs(CLEAN, exist_ok=True)

    deadline = Deadline(args.timeout_seconds)
    shards = list_train_shards()
    print(f"[clean] {len(shards)} train shards, timeout={args.timeout_seconds}s")

    # Pass 1: labels + dedup (cheap, always completes).
    labels, dup_drop, n_total, class_counts, order = pass1_labels_and_dedup(shards)
    print(f"[clean] pass1 done: n_total={n_total} duplicates={len(dup_drop)} "
          f"({deadline.elapsed():.1f}s)")

    # Pass 2: metadata + deterministic cleaning (deadline-bounded).
    meta, drop_reason, montage, n_examined, stopped_early = \
        pass2_metadata_and_clean(shards, labels, dup_drop, deadline)
    print(f"[clean] pass2 done: examined={n_examined} "
          f"corrupt+degenerate={len(drop_reason)} "
          f"stopped_early={stopped_early} ({deadline.elapsed():.1f}s)")

    # Summary + figures + manifest (best effort, always attempted).
    summary = build_summary(
        n_total, class_counts, dup_drop, drop_reason, labels, meta,
        n_examined, stopped_early, args.timeout_seconds, deadline.elapsed())
    with open(os.path.join(TASK01, "exploration_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    try:
        make_figures(summary, labels, drop_reason, meta, montage)
    except Exception as exc:  # figures are non-critical
        print(f"[clean] WARNING: figure generation failed: {exc}")

    n_manifest = write_manifest(shards, labels, dup_drop, drop_reason, meta, order)

    d = summary["drops"]
    print(f"[clean] DONE total={n_total} kept={summary['n_kept']} "
          f"manifest_rows={n_manifest} "
          f"dropped_dup={d['duplicates']} dropped_corrupt={d['corrupt']} "
          f"dropped_degenerate={d['degenerate_min_side'] + d['degenerate_aspect']} "
          f"({deadline.elapsed():.1f}s)")


if __name__ == "__main__":
    main()
