# Task 1.1 — Dataset Exploration & Cleaning (`clean.py`) — Notes

This document explains what `solution/clean.py` does, **why** each decision was
made, and how to run and verify it. Task 1.1 is worth 15/100 points and asks for
three things: (1) explore the dataset (class distribution, image-size
distribution, descriptive stats), (2) report which characteristics let you deduce
the class, and (3) build a **deterministic** cleaning pipeline — justifying the
choices rather than applying a fixed recipe. This is cleaning, **not**
augmentation.

---

## 1. What the data looks like

- **Location:** `solution/data/train/` — 7 parquet shards (`train_0..6.parquet`),
  ~866 MB total, **29,688 rows**.
- **Schema:** `image: binary` (compressed JPEG/PNG bytes), `source_class: int8`.
- **Label mapping:** the exercise is binary — `real (0)` vs `ai_generated (1)`.
  The raw `source_class` has 6 values; class `0` is real and classes `1..5` are
  different AI generators that we **merge into label 1**
  (`to_label = 0 if source_class == 0 else 1`).
- **Class balance:** ~**16.7 % real / 83.3 % AI**. Real is the minority class —
  important later because Task 1.2 measures false-positive rate *on the real
  class*.

### The headline finding: a trivial "shortcut"

Measured over the examined rows:

| class       | % square | median min-side | median aspect | median file size |
| ----------- | -------- | --------------- | ------------- | ---------------- |
| 0 real      | 3.2 %    | 428 px          | 1.34          | 47.3 KB          |
| 1 ai        | 100 %    | 320 px          | 1.00          | 25.1 KB          |

AI images are **uniformly 320×320 squares and smaller in both pixels and
bytes**; real images are larger and non-square. That means a model could "cheat"
by reading raw size / aspect-ratio / file-size metadata instead of learning
anything about image content. This is the single most report-worthy observation
and it drives the central cleaning decision below.

---

## 2. Design decisions (and the reasoning)

### 2.1 Output a *manifest*, not a materialized copy of cleaned images

Re-emitting the cleaned images would duplicate ~866 MB and blow the 20 MB
submission-zip limit, and `data/` is a **read-only** mount so we cannot write
back there anyway. Instead `clean.py` writes a lightweight
`clean_manifest.parquet` listing the **kept rows** — `(shard_file, row_index,
label, width, height, n_bytes, examined)`. `prepare.py` (Task 1.2) replays this
manifest against the read-only `data/` to reconstruct the cleaned set on the fly.
This satisfies the exercise's "a script that regenerates the cleaned dataset"
option while keeping artifacts tiny (the manifest is ~119 KB).

### 2.2 Stream shard-by-shard to bound RAM

The whole dataset never fits comfortably in memory alongside decoded images, so
we read with `pyarrow.ParquetFile.iter_batches(batch_size=512)` and process one
batch at a time. Per-pixel statistics are computed on a **downscaled copy**
(longest side ≤ 256 px) so cost is bounded by the analysis size, not the original
resolution.

### 2.3 Cleaning is deterministic and content-preserving

We drop a row **only** for one of these objective reasons:

1. **Exact duplicate** — SHA1 of the raw image bytes; keep the first occurrence
   in deterministic shard/row order. (Found **312** exact duplicates.)
2. **Corrupt / undecodable** — PIL fails to open/convert/load the bytes.
3. **Degenerate geometry** — `min(w, h) < 32 px` (too small to be useful) or
   `aspect ratio > 5:1` (pathological strips, almost certainly junk).

These thresholds are justified by the size distribution, not picked arbitrarily,
and they're recorded in the summary so the choices are auditable.

### 2.4 We deliberately do NOT clean away the shortcut

It is tempting to "fix" the leakage by, say, dropping AI images until the size
distributions match. **We don't**, because that would distort the label
distribution and throw away most of the data. Instead the *cleaning step
documents* the shortcut and records a downstream **128×128 RGB resize target**.
Resizing every image to a fixed square in `prepare.py` removes the
size/aspect/byte signal entirely, forcing the model to learn image **content** —
which is exactly what makes it robust to the augmented test set in Task 1.3. So
the mitigation lives in preprocessing; cleaning only quantifies and records it.

### 2.5 Timeout-aware, written best-effort

The grader runs `python clean.py --timeout_seconds 600` and **kills the process**
at the limit. A small `Deadline` helper (monotonic clock, 0.9 safety factor)
governs the expensive work:

- **Pass 1** (labels + dedup) is cheap and **always completes** (~2 s).
- **Pass 2** (decode + metadata + drop rules) runs until the deadline. Any rows
  not reached are **kept** and flagged `examined=False` in the manifest (only
  their byte-duplicates were already removed). This way a short timeout shrinks
  the *analysis*, never the *kept training set* — a generous 600 s timeout
  examines everything (`examined_all = true`).

### 2.6 Determinism

`random.seed(0)` / `np.random.seed(0)`, sorted shard iteration, and stable SHA1
hashing mean repeated runs produce identical dedup/drop counts and manifest row
counts.

---

## 3. Code structure (`solution/clean.py`)

| Section | Responsibility |
| ------- | -------------- |
| Constants/paths | `DATA_DIR`, `TASK01`, `CLEAN` dirs; `BATCH_SIZE=512`, `ANALYSIS_MAX=256`, `RESIZE_TARGET=128`, drop thresholds; `to_label()` |
| `Deadline` | monotonic budget with safety factor; `expired()`, `remaining()`, `elapsed()` |
| IO/decode | `list_train_shards()` (sorted), `iter_batches()`, `decode_rgb()` (PIL → RGB, returns `None` on failure), `image_meta()` (w/h/min_side/aspect/megapixels/bytes/square + brightness/contrast/saturation on the downscaled copy) |
| `pass1_labels_and_dedup()` | streams all shards; returns labels, duplicate set, totals, per-class counts, deterministic row order |
| `pass2_metadata_and_clean()` | streams again; decodes, computes metadata, applies drop rules; collects a few thumbnails per class for the montage; stops at the deadline |
| `build_summary()` | aggregates per-class percentile stats, drop counts, and the `shortcut_findings` block into a JSON-serializable dict |
| `make_figures()` | the 5 PNG figures (Agg backend) |
| `write_manifest()` | writes `clean_manifest.parquet` of kept rows via pyarrow |
| `main()` | CLI, seeds, dir creation, orchestrates passes, writes outputs, prints a one-line summary |

---

## 4. Outputs

All written under `solution/artifacts/` (the only writable location):

- `artifacts/task01/exploration_summary.json` — class distribution, per-class
  size/aspect/byte/pixel percentiles, drop counts + rules, the
  `shortcut_findings` block, and the chosen `downstream_resize_target`.
- `artifacts/task01/class_balance.png` — real vs AI counts.
- `artifacts/task01/size_scatter.png` — width vs height colored by class
  (visually shows the AI 320×320 cluster vs the spread-out real images).
- `artifacts/task01/aspect_hist.png` — aspect-ratio distribution per class
  (AI is a spike at 1.0).
- `artifacts/task01/bytes_hist.png` — encoded file-size distribution per class.
- `artifacts/task01/montage.png` — example thumbnails (top row real, bottom AI).
- `artifacts/clean/clean_manifest.parquet` — the kept rows for `prepare.py`.

---

## 5. How to run

```bash
cd solution
python clean.py --timeout_seconds 600
```

Needs only `pandas`/`pyarrow`/`numpy`/`PIL`/`matplotlib` — **no torch** — so it
runs on the default interpreter.

---

## 6. Verification performed

- `python -m py_compile clean.py` → compiles clean.
- **Smoke run** `--timeout_seconds 60`:
  - `n_total = 29688`, `duplicates = 312`, `kept = 29376`.
  - Pass 2 examined 7,201 rows then stopped early (`stopped_early_due_to_timeout
    = true`); the manifest still has all 29,376 kept rows (7,201 `examined=True`
    + 22,175 `examined=False`). This confirms the timeout degradation keeps the
    training set intact.
  - All 5 PNGs + the JSON summary + the manifest were written.
- Determinism: the dedup count (312) is identical across runs and matches earlier
  data copies.
- Read-only compliance: the code only ever **reads** from `data/` and writes
  exclusively under `artifacts/`.

A full 600 s run examines every row (`examined_all = true`) and reproduces the
real-vs-AI size/aspect/byte separation in the summary.

---

## 7. What's intentionally out of scope here

Augmentation, model training, and the fixed-square resize itself belong to later
tasks (`prepare.py`, `train.py`, …). `clean.py` only explores, cleans
deterministically, and records the resize target so downstream steps stay
consistent. A shared `common.py` can be factored out when `prepare.py` is built;
`clean.py` is self-contained for now.
