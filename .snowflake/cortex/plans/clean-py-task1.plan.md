---
name: "clean-py-task1"
created: "2026-06-07T18:32:56.842Z"
status: pending
---

# Plan: Implement solution/clean.py (Task 1.1 - Exploration & Cleaning, 15 pts)

## Context

This is a fresh project at `c:\Users\OnurDilsiz\Master\sose2026\amls_project`. `solution/clean.py` is currently **empty**, and `solution/data/` already contains the real dataset.

Verified facts from exploration:

- **Schema** (`data/train/*.parquet`): `image: binary` (compressed JPEG/PNG bytes), `source_class: int8`. 7 train shards, `train_0` = 4242 rows, ~~29.7k rows total (~~866 MB).

- **Environment**: default Python 3.12.10 has `pandas 3.0.0`, `pyarrow 23.0.0`, `numpy 2.4.1`, `PIL 12.1.1`, `matplotlib 3.10.8` — all working. `torch` is broken on the default interpreter, but **clean.py needs no torch** (pure data/image-stats work).

- **Dataset shortcut confirmed** (800-image sample of `train_0`):

  | class  | share | avg WxH | avg KB | % square | avg aspect |
  | ------ | ----- | ------- | ------ | -------- | ---------- |
  | 0 real | \~16% | 583x466 | 47.2   | 5%       | 1.43       |
  | 1 ai   | \~84% | 310x310 | 25.7   | 100%     | 1.00       |

  AI images are uniformly square and smaller in pixels and bytes; real images are larger and non-square. This is a **trivial raw-metadata shortcut** (size / aspect / file-bytes), which is the headline finding the report must call out.

Submission constraints (from the PDF) that clean.py must honor:

- `data/` is a **read-only** mount; write everything under `solution/artifacts/`.
- Run as `python clean.py --timeout_seconds 600`; the process is killed at timeout, so write outputs incrementally / best-effort.
- Must be **deterministic** and re-runnable. Deliverable = runnable code + a cleaned dataset *or a script that regenerates it* + a short visual findings report.

## Design decisions

1. **Output a manifest, not a materialized copy.** Re-emitting cleaned images would duplicate \~866 MB and blow the 20 MB zip limit. Instead clean.py writes `artifacts/clean/clean_manifest.parquet` listing the kept rows `(shard_file, row_index, label, width, height, n_bytes)`. `prepare.py` (later task) replays this manifest against the read-only `data/`. This satisfies "a script that regenerates it" and keeps artifacts tiny.

2. **Stream shard-by-shard** with pyarrow `ParquetFile` to bound RAM (never load all 866 MB at once). Cap per-image pixel analysis to a downscaled copy (longest side <= 256) so brightness/contrast/saturation stats stay cheap.

3. **Cleaning is deterministic and content-preserving** (this task is cleaning, NOT augmentation):

   - **Dedup**: drop exact-duplicate images via SHA1 of raw bytes (prior runs found \~312 dups). Keep first occurrence in deterministic shard/row order.
   - **Drop undecodable/corrupt** images (PIL fails to open/verify).
   - **Drop degenerate** images: `min(w,h)` below a small threshold (e.g. < 32) or pathological aspect ratio (e.g. > 5). These are justified by the size distribution, not arbitrary.
   - **Do NOT drop** on the shortcut features (squareness, byte size). Removing them would distort the label distribution. The recipe instead **documents** that downstream preprocessing resizes every image to a fixed square (e.g. 128x128 RGB) which *neutralizes* the size/aspect/bytes shortcut and forces the model to learn content. clean.py records the chosen target so prepare.py is consistent.

4. **Timeout-aware**: a small `Deadline` helper (monotonic clock, \~0.9 safety factor). The label/dup pass is cheap and runs fully; the per-image pixel-stat pass runs until the deadline and degrades gracefully (rows not yet examined are still kept, only marked unexamined). Summary + figures are written at the end of whatever was reached.

5. **Determinism**: fixed `random.seed(0)` / `np.random.seed(0)` for any sampling (e.g. montage selection), sorted shard iteration, stable hashing.

## Implementation steps

1. **Scaffold + CLI**: argparse `--timeout_seconds` (default 600). Define paths (`DATA_DIR=data`, `ART=artifacts`, `TASK01=artifacts/task01`, `CLEAN=artifacts/clean`), create artifact dirs, set seeds, force matplotlib `Agg` backend.

2. **Deadline + IO helpers**: `Deadline` class; `iter_train_shards()` yielding `(shard_name, pyarrow_table)` one shard at a time; `decode(img_bytes)` -> PIL RGB (with try/except); `image_meta(im, raw_bytes)` -> dict of `w,h,aspect,min_side,megapixels,n_bytes,mode` plus cheap pixel stats (mean/std brightness, per-channel mean, saturation) on a <=256px downscaled copy.

3. **Pass 1 - labels + dedup (cheap, always completes)**: stream all shards reading only `source_class` + hashing `image` bytes. Accumulate `n_total`, per-class counts, `seen_hashes` set, and a per-row `keep` flag (False for byte-duplicates).

4. **Pass 2 - metadata + cleaning decisions (deadline-bounded)**: stream shards again; for each not-yet-dropped row, decode + compute `image_meta`, apply the deterministic drop rules (undecodable, min\_side, aspect), and record metadata. Stop the per-image work at the deadline; rows past that point stay kept+unexamined.

5. **Exploration summary**: aggregate into `artifacts/task01/exploration_summary.json` — class distribution (count + %), size/aspect/byte distributions per class (min/median/mean/max + percentiles), `% square` per class, dedup count, drop counts by reason, and an explicit `shortcut_findings` block (the size/aspect/bytes separation) plus the chosen downstream resize target and justification text.

6. **Figures** (matplotlib -> PNG in `artifacts/task01/`): (a) class-balance bar, (b) width-vs-height scatter colored by class, (c) aspect-ratio histogram per class, (d) file-bytes histogram per class, (e) a small example montage (a few real + a few AI thumbnails). Each saved with `dpi<=110` to stay small.

7. **Write manifest**: `artifacts/clean/clean_manifest.parquet` of kept rows `(shard_file, row_index, label, width, height, n_bytes, examined)`, written via pyarrow. Print a concise stdout summary (`total / kept / dropped_dup / dropped_corrupt / dropped_degenerate`).

8. **Self-review pass**: re-read clean.py for read-only compliance (no writes under `data/`), determinism, and graceful timeout behavior; run `python -m py_compile clean.py` and a short `--timeout_seconds 60` smoke run to confirm artifacts appear.

## Verification

- `python -m py_compile solution/clean.py` compiles clean.
- Smoke run: `cd solution; python clean.py --timeout_seconds 60` — completes without error, writes `artifacts/task01/exploration_summary.json` + PNGs and `artifacts/clean/clean_manifest.parquet`; stdout shows total≈29.7k and a nonzero dedup count.
- Full run: `python clean.py --timeout_seconds 600` examines all rows (manifest `examined` all true) and reproduces the real-vs-AI size/aspect/bytes separation in the summary.
- Determinism check: run twice; manifest row count, dedup count, and drop counts are identical.
- Confirm nothing was written under `solution/data/` (read-only respected).

## Critical files

- `solution/clean.py` - the entire deliverable for Task 1.1 (currently empty; to be written).
- `AMLS_2026_Exercise.pdf` - task spec (Task 1.1 requirements, read-only `data/`, artifacts layout, `--timeout_seconds` contract).
- `solution/data/train/*.parquet` - read-only input streamed for exploration + cleaning.
- `solution/artifacts/task01/` + `solution/artifacts/clean/` - output locations (created at runtime).

## Out of scope (later tasks, not this plan)

prepare.py / train.py / predict.py (Task 1.2), train\_augmented.py / predict\_augmented.py (Task 1.3), explainability (Task 1.4), Dockerfile, and the PDF report. A shared `common.py` of helpers can be factored out when building prepare.py; clean.py stays self-contained for now.
