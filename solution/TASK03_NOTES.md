# Task 1.3 - Data Augmentation and Feature Engineering (30 pts)

## Goal
Make the detector robust to scaled / compressed / blurred / noised images, reaching
`recall_ai >= 0.6` on `data/validation_augmented/` while keeping `FPR_real <= 20%`,
continuing from the Task 2 starting point.

## Deliverables (in `solution/`)
- `train_augmented.py --timeout_seconds 1800` - trains the robust model, recalibrates on
  `calibration_augmented`, gates on `validation_augmented`, persists the winner to `artifacts/task03/`.
- `predict_augmented.py --timeout_seconds 600` - runs the winner on `data/predict/` ->
  `artifacts/task03/predictions.csv` (same format as Task 2: `row_id,predicted_label`).
- `common.augment_u8(img, rng)` - one random augmentation per call: JPEG recompression (q 35-85),
  Gaussian blur (r 0.5-1.5), downscale-then-upscale (0.5-0.85x), or additive Gaussian noise.

## Approach
1. **Augmented training set**: each training image is augmented once with `augment_u8`; the
   leakage-safe 57-dim features are recomputed on the augmented copy and stacked with the cached
   clean features (29,376 clean + 29,376 augmented = 58,752 rows). The standardiser is refit on this
   distribution. This teaches the classical models the perturbation-invariant feature regions.
2. **CNN continuation**: the Task 2 CNN checkpoint (`artifacts/task02/cnn_best.pt`) is fine-tuned on
   augmented 64px batches (~70% of each batch augmented on the fly).
3. **Operating point**: threshold recalibrated on `calibration_augmented` (target FPR 19%), hard 20%
   gate verified on `validation_augmented`. A weighted ensemble is swept on the augmented calibration
   split. The winner is the family with the best `validation_augmented` recall_ai under FPR <= 20%.

## Results (Docker-validated, `--cpus 8`, full 1800s budget)
| Operating point | recall_ai | FPR_real | gate |
|---|---|---|---|
| Task 2 model on `validation` (clean) | 0.808 | 0.192 | PASS |
| Task 2 model on `validation_augmented` (Task-2 threshold) | 0.643 | 0.337 | FAIL |
| Task 3 classical (histgb) on `validation_augmented` | 0.482 | 0.171 | PASS |
| Task 3 CNN on `validation_augmented` | 0.577 | 0.134 | PASS |
| **Task 3 ensemble (winner, w_clf=0.4)** on `validation_augmented` | **0.615** | **0.177** | **PASS** |
| Task 3 ensemble (winner) on `validation` (clean) | 0.830 | 0.197 | PASS |

- **Winner = ensemble** (`0.4*HistGB + 0.6*CNN`): `validation_augmented` recall_ai **0.615**
  at FPR 0.177 — **meets the 0.6 aim** while respecting the 20% gate, and stays strong on clean
  `validation` (recall 0.830). Confusion (vaug): tp576/fn361/tn154/fp33.
- Two changes unlocked this: (1) **BatchNorm in the CNN** (`common.make_cnn`) so the from-scratch
  CNN actually converges on CPU, and (2) an **anti-overfitting fine-tune** of that CNN on augmented
  data (LR 2e-4 + decay, weight decay, 50% augmented batches) so it adapts to perturbations without
  memorising them.
- **Calibration detail (report-worthy):** on augmented data the calibration→validation FPR gap runs
  *conservative* (vaug FPR lands below the calibration target), so we calibrate the operating point
  at the full 20% budget on `calibration_augmented`; the hard gate is then verified on
  `validation_augmented` (0.177 ≤ 0.20). This is the mirror image of Task 2, where the gap runs the
  other way and we calibrate below 20%.

## Key finding (report-worthy)
The augmentations that hurt most - **blur, JPEG compression and downscaling - directly destroy the
high-frequency spectral fingerprint** that the Task 2 classical model relies on. Consequently, at a
strict `FPR <= 20%` operating point the classical recall on augmented data falls from ~0.69 (clean)
to ~0.47. The **CNN learns lower-frequency spatial/content cues that survive these perturbations**,
so it is the augmentation-robust component and the ensemble leans on it. This is exactly the kind of
shortcut/robustness trade-off the brief asks to discuss: the spectral cue is powerful on clean data
but fragile, and genuine robustness comes from content-level features.

## Run order (Task 3)
```
python train_augmented.py   --timeout_seconds 1800
python predict_augmented.py --timeout_seconds 600
```
