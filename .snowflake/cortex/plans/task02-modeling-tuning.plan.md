# Plan: Task 1.2 — Modeling and Tuning under Time Constraints (35 pts)

## Goal
Build a reproducible, CPU-only, internet-free ML pipeline that classifies images as `real(0)` / `ai_generated(1)`, **maximizing `recall_ai` subject to `FPR_real <= 20%`** (calibrated on `data/calibration/`, verified on `data/validation/`). Train and compare **two model families** (classical baseline on engineered features + CNN from scratch). Ship the single best pipeline as `prepare.py` + `train.py` + `predict.py`. Target `recall_ai >= 0.8`.

## Confirmed inputs (already verified)
- `artifacts/clean/clean_manifest.parquet` — 29,376 kept train rows (cols: `shard_file,row_index,label,width,height,n_bytes,examined`).
- Labeled splits (cols `image:binary, source_class:int8`): train (7 shards), calibration (1,924), validation (1,124), validation_augmented (1,124), calibration_augmented (1,924).
- predict (100 rows, cols `row_id:int32, image:binary`).
- `C:/amlsvenv` Python has `torch 2.5.1+cpu`, pyarrow, PIL, numpy, sklearn.
- Shortcut from Task 1.1: AI = 100% square 320px / 25 KB; real = 3% square / 428px / 47 KB. **Square-resize to 128x128 neutralizes the size shortcut; raw size/bytes must NOT be used as model features.**

## Design principles
- **Read-only `data/`**: write everything only under `artifacts/`.
- **Timeout-aware**: reuse `Deadline` helper (0.9 safety); checkpoint best model frequently so a kill still leaves a usable artifact.
- **Reproducible**: fix seeds (numpy, torch); `torch.set_num_threads(min(8,cpu))`, `set_num_interop_threads(1)` (matches Appendix C, --cpus 8 grading).
- **No predict prep**: `prepare.py` must NOT touch `data/predict/` (it changes at grading). `predict.py` does its own decode+resize inline.
- **Leakage-safe features**: exclude width/height/bytes/aspect; rely on content/frequency artifacts only so the model generalizes to the hidden holdout.

## New files
1. `solution/common.py` — shared helpers (imported by prepare/train/predict; container runs from same folder).
2. `solution/prepare.py`
3. `solution/train.py`
4. `solution/predict.py`

---

## Step 1 — `common.py` (shared contract)
- `SEED=0`, `RESIZE=128`, paths: `DATA_DIR`, `ART`, `PREP=artifacts/prepared`, `TASK02=artifacts/task02`, `CLEAN_MANIFEST=artifacts/clean/clean_manifest.parquet`.
- `class Deadline` (copy from clean.py: monotonic start, budget=timeout*0.9, `remaining/expired/elapsed`).
- `to_label(source_class) -> 0 if int(sc)==0 else 1`.
- `decode_resize(raw_bytes) -> np.uint8 [128,128,3]`: PIL open -> convert RGB -> `resize((128,128), BILINEAR)` -> np array; return None on failure.
- `engineered_features(img_uint8) -> np.float32[D]`: leakage-safe, ~30–40 dims:
  - per-channel mean/std/skew; overall brightness/contrast/saturation;
  - edge density via Laplacian/gradient magnitude;
  - **FFT high-frequency energy ratio** (AI images have characteristic spectral signatures), radial power-spectrum bands;
  - residual/noise stats (img minus 3x3 blur) mean/std/kurtosis;
  - per-channel correlation, color histogram entropy.
  - (NO width/height/bytes/aspect.)
- `make_cnn(k=32)`: Appendix-B architecture — `Conv(3,k)->ReLU->MaxPool -> Conv(k,2k)->ReLU->MaxPool -> Conv(2k,4k)->ReLU -> AdaptiveAvgPool2d(1) -> Flatten -> Linear(4k,2)`. Parameterize `k`.
- `set_determinism()`, `set_threads()`.
- `iter_split(split)`: stream parquet batches yielding `(raw_bytes, source_class)` (or `row_id` for predict).

## Step 2 — `prepare.py` (`--timeout_seconds` default 600)
- Replay `clean_manifest` to select kept train rows per shard (group manifest by `shard_file`, gather `row_index` sets + labels).
- For each split in **{train, calibration, validation, validation_augmented, calibration_augmented}** (NOT predict):
  - Stream shards; for train, keep only manifest rows; decode+resize to 128x128 uint8; compute `engineered_features`; collect labels.
  - Write to `artifacts/prepared/<split>_images.npy` (uint8, N×128×128×3, via memmap to bound RAM), `<split>_feats.npy` (float32 N×D), `<split>_labels.npy` (int8). predict has no labels and is skipped.
- Standardization stats: fit feature `mean`/`std` on **train only**, save `feature_scaler.npz`; apply at train/predict time (don't leak val/calib stats).
- Deadline-aware: process train first (largest); write incrementally; if time runs short, persist whatever completed (record `prepared_summary.json` with per-split counts + `complete` flag).
- Output: `artifacts/prepared/*` + `prepared_summary.json`.

## Step 3 — `train.py` (`--timeout_seconds` default 1800)
- Load prepared train (images+feats+labels), calibration, validation, validation_augmented.
- Handle class imbalance (train ~16.7% real): class-weighted loss / weighted sampler for CNN; `class_weight='balanced'` for classical.
- **Family A — classical baseline**: sklearn on standardized engineered features (LogisticRegression + GradientBoosting/RandomForest); pick best by val recall_ai@FPR<=20%. Fast (seconds).
- **Family B — CNN from scratch** (Appendix-B, pick `k` to respect time budget; start k=32, fall back to 24/16 if needed):
  - AdamW lr=1e-3, CrossEntropyLoss (weighted), batch 128, BILINEAR-resized float tensors normalized to [0,1] (or per-channel mean/std).
  - Epoch loop bounded by `Deadline`; after each epoch evaluate on calibration+validation; **save best checkpoint** (by val recall_ai subject to FPR<=20%) to `artifacts/task02/cnn_best.pt` immediately (survives timeout kill).
  - Log per-epoch metrics.
- **Threshold calibration (automatic, both families)**: on `data/calibration/`, sweep threshold on `P(ai)`; choose the **most permissive threshold with FPR_real <= target** where `target=0.18` (margin below 0.20 to stay safe on hidden holdout). This maximizes recall_ai while respecting the constraint.
- **Validation protocol**: at calibrated threshold, compute on `validation` and `validation_augmented`: FPR_real, recall_ai, precision, accuracy, confusion matrix. Verify `FPR_real <= 0.20` on validation (the hard gate).
- **Model selection**: pick the family with higher validation recall_ai that satisfies FPR<=20%. Persist winner to `artifacts/task02/`:
  - `model_meta.json` (family, k, threshold, feature columns, metrics on val + val_aug),
  - CNN: `cnn_best.pt`; classical: `classical.joblib` + `feature_scaler.npz`.
- Reproducibility + timing note: print a one-line metrics summary; record `train_report.json`.

## Step 4 — `predict.py` (`--timeout_seconds` default 600)
- Load `model_meta.json` + winning model + threshold.
- Stream `data/predict/*.parquet` (`row_id,image`); for each: `decode_resize`; for classical also `engineered_features`+scaler; run inference -> `P(ai)`; `predicted_label = 1 if P(ai) >= threshold else 0`.
- Robust to unreadable images (fallback label 0 = conservative on false accusations).
- Write `artifacts/task02/predictions.csv` with header `row_id,predicted_label`, sorted by row_id.

## Step 5 — Validate end-to-end (smoke)
- Run `prepare.py --timeout_seconds 120` (partial ok), `train.py --timeout_seconds 180` (few epochs), `predict.py --timeout_seconds 120` using `C:/amlsvenv`.
- Confirm: prepared arrays exist; CNN checkpoint + threshold saved; `predictions.csv` has 100 rows + correct header; validation FPR_real <= 0.20 reported.
- Then a longer real run for quality (train near full 1800s budget) before reporting numbers.

## Out of scope (later tasks)
- `train_augmented.py` / `predict_augmented.py` (Task 1.3), explainability (Task 1.4), `Dockerfile` + `requirements.txt`, `report.pdf`.

## Deliverables of this step
- `solution/common.py`, `solution/prepare.py`, `solution/train.py`, `solution/predict.py`
- Runtime artifacts: `artifacts/prepared/*`, `artifacts/task02/{model_meta.json, cnn_best.pt|classical.joblib, predictions.csv, train_report.json}`
- `TASK02_NOTES.md` documenting model comparison, calibration, and validation metrics (consistent with the clean.py notes style).