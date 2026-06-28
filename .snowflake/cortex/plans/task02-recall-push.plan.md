# Plan: Push Task 1.2 recall_ai toward 0.8

## Goal
Raise validation `recall_ai` from **0.593** (current classical gboost winner) toward **0.8**, while keeping the hard gate `FPR_real ≤ 0.20`. Diagnosis: the from-scratch CNN reached only **0.417** because just **2 epochs** fit the 1800s budget at 128×128. Giving it a 64px view quarters per-step cost (~4× more epochs) and lets it actually learn content.

## Key design decisions
- **Reuse existing 128px caches** (`artifacts/prepared/*_images.npy`). Engineered features depend on RESIZE=128, so caches stay as-is. The CNN simply gets a **downsampled 64×64 view** computed on-the-fly in the batch loader. **No prepare.py rerun** (saves ~600s, avoids re-decoding 29k images).
- `make_cnn` is unchanged — `AdaptiveAvgPool2d(1)` makes it resolution-agnostic, so the same Appendix-B architecture works at 64px. **No pretrained weights** (compliant).
- Runtime stays far under the 5× Appendix-C budget (we use the shallower Appendix-B net at lower res).
- Determinism preserved: downsample is exact; flip augmentation uses the seeded RNG.

## Changes by file

### 1. `common.py`
- Add `CNN_RES = 64`.
- Add `downsample_u8(batch_u8, src=128, dst=64)`: exact 2×2 block-mean via reshape `(N, 64, 2, 64, 2, 3).mean(axis=(2,4))` → uint8. Deterministic, no PIL.

### 2. `train.py` — CNN family rework
- **64px loader**: in `train_cnn` and `cnn_scores`, downsample each uint8 chunk to 64px before building the tensor.
- **Throughput**: batch size 128 → **256** (bigger matmuls are more CPU-efficient).
- **Milder class weighting**: replace the current inverse-frequency weight (over-pushes toward "real", hurting recall) with a damped `sqrt(inverse-freq)`. Threshold calibration already handles the FPR trade-off; the loss should mainly produce well-ranked probabilities.
- **Light regularization** (helps clean-validation generalization, *not* the Task 1.3 robustness work): seeded random horizontal flip per batch.
- **LR decay**: AdamW lr=1e-3, drop to 3e-4 once `deadline.elapsed()` passes ~60% of budget (cheap, stabilizes late epochs).
- Per-epoch checkpoint of best-by-recall@FPR≤0.20 is **kept** (timeout-safe).

### 3. `train.py` — ensemble candidate
- After both families train, form `p_ens = 0.5*(p_cnn + p_classical)` on the **calibration** split, calibrate a threshold, evaluate on **validation**.
- Winner selection extends to three candidates — `classical`, `cnn`, `ensemble` — by the existing rule (max `recall_ai` among those with `fpr_real ≤ 0.20`).
- If `ensemble` wins, persist `family="ensemble"` in `model_meta.json` and ship **both** `classical.joblib` + `feature_scaler.npz` + `cnn_best.pt` with a single threshold on the averaged probability.

### 4. `predict.py`
- CNN branch: downsample decoded 128px array to 64px before the tensor (mirror training).
- New `family == "ensemble"` branch: load both models, average `P(ai)`, threshold once. Unreadable image → label 0 fallback unchanged.

### 5. Run & verify (artifact-driven, per host lessons)
- `python train.py --timeout_seconds 1800` (background shell + log polling; judge by artifacts, never queue destructive cmds).
- Then `python predict.py --timeout_seconds 600`.
- Check `train_report.json` / `model_meta.json`: confirm winner, `recall_ai`, and **`fpr_real ≤ 0.20` gate = PASS**. Confirm `predictions.csv` = 100 rows + header.

### 6. Document
- Refresh `TASK02_NOTES.md` (new CNN-at-64px rationale, ensemble, final numbers) and update `/memories/amls-ai-image-detection.md`.

## Success criteria
- Hard gate holds: `fpr_real ≤ 0.20` on validation (non-negotiable).
- `recall_ai` materially above 0.593; target ~0.8 (stretch — reported honestly whatever it lands at; if CNN/ensemble don't beat classical under the gate, classical stays the winner and nothing regresses).
- Pipeline still reproducible, CPU-only, internet-free, within timeouts.

## Out of scope
- Task 1.3 augmentation/robustness, Task 1.4 explainability, Dockerfile/requirements.txt/report.pdf (tracked separately).