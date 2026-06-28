# AMLS 2026 — AI Image Detection: Development Log & Retrospective

> Internal document for the team. **Not part of the submission zip** (it lives at repo
> root next to `report_build.py` / `report.md`, which are also dev-only). Use it to
> understand the whole development process: what we built, what went wrong, what fixed
> it, and which changes actually moved the numbers.

---

## 0. Start here — how to understand this repo (for the team)

Read/run things in this order. Budget ~1–2 hours for a full pass.

**Step 1 — Read, in this order (no code yet):**
1. `AMLS_2026_Exercise.pdf` — the assignment itself. Focus on pages 1–3 (submission
   rules, the 6-script execution order + timeouts, and the 0.8 recall / 20% FPR aim).
2. This file (`DEVELOPMENT.md`) top to bottom — the *story* of how we got there.
3. `report.md` — the polished version of what we'd tell the graders.

**Step 2 — Understand the pipeline by reading code in execution order:**
Open these in the same order the grader runs them. Each script imports `common.py`,
so read `common.py` *first* and keep it open as a reference.
1. `solution/common.py` — constants (SEED, RESIZE, MAX_FPR, CALIB_TARGET_FPR…),
   `engineered_features` (the 57-dim feature vector), `make_cnn` (BatchNorm CNN),
   `calibrate_threshold`, `augment_u8`. **This is the brain of the project.**
2. `solution/clean.py` → loads/validates/splits raw data.
3. `solution/prepare.py` → builds feature matrices + tensors.
4. `solution/train.py` → Task 1.2: classical models + CNN + weighted ensemble.
5. `solution/predict.py` → Task 1.2: writes `artifacts/task02/predictions.csv`.
6. `solution/train_augmented.py` → Task 1.3: robustness under augmentation.
7. `solution/predict_augmented.py` → Task 1.3: writes `artifacts/task03/predictions.csv`.
8. `solution/explain.py` → Task 1.4: feature importance, saliency, FP/FN analysis.

**Step 3 — Run it yourself (CPU-only, no internet). Two options:**
- *Native (fastest to iterate):* create a venv, `pip install -r solution/requirements.txt`,
  put the data under `solution/data/`, then run the 7 scripts in the order above from
  inside `solution/`. Watch `artifacts/task02/model_meta.json` and
  `artifacts/task03/model_meta.json` for the recall/FPR numbers.
- *Docker (matches the grader exactly):* `docker build` the `solution/Dockerfile`, then
  run each script in the container. **Use this to reproduce the official numbers** —
  local sklearn/torch versions differ from the image and won't unpickle Docker-trained
  models. See §8 (environment gotchas) for the crashes/quirks we hit.

**Step 4 — Connect results to decisions:** with the numbers in front of you, read
§5 ("what moved the needle") and §6 ("what we did wrong"). That's where the *why*
behind every constant in `common.py` lives.

**The 3 things to internalize if you read nothing else:**
- **BatchNorm in the CNN** was the single biggest jump (Appendix B has none).
- **Residual-spectrum FFT features** are leakage-safe and catch diffusion fingerprints.
- **Calibration is direction-dependent**: calibrate *below* 20% on clean data (0.17),
  *at* 20% on augmented data — see §5/§6 for why.

---

## 1. The assignment in one paragraph

Build a fully reproducible, CPU-only, internet-free ML pipeline that detects AI-generated
images (binary: `0=real`, `1=ai_generated`; source classes 1–5 merged into 1). The grader
mounts `data/` read-only, runs six scripts in a fixed order with per-script timeouts, and
scores predictions on a **hidden holdout**. Pass = ≥50/100; ≥90 = +5 exam points. Two hard
targets:

- **Task 1.2** (35 pts): maximize `recall_ai` with **FPR_real ≤ 20%** on `validation`. Aim ≥ **0.80**.
- **Task 1.3** (30 pts): same constraint on `validation_augmented`. Aim ≥ **0.60**.

Plus Task 1.1 (cleaning, 15 pts) and Task 1.4 (explainability, 20 pts).

---

## 2. Final results (what we achieved) — Docker-validated

| Task | Metric | Result | Gate | Status |
|------|--------|--------|------|--------|
| 1.2  | recall_ai @ FPR | **0.8077 @ 0.1915** | ≥0.80, FPR≤0.20 | PASS (tp756/fn180/tn152/fp36) |
| 1.3  | recall_ai @ FPR (val_aug) | **0.6147 @ 0.1765** | ≥0.60, FPR≤0.20 | PASS (tp576/fn361/tn154/fp33) |
| 1.3  | recall_ai @ FPR (val, clean) | 0.8301 @ 0.1968 | FPR≤0.20 | within budget |

Both aims met, both strictly under the 20% false-positive cap. Winning model in both tasks:
a **weighted ensemble** of a classical engineered-feature model + a from-scratch CNN.

---

## 3. Pipeline architecture

```
clean.py            # Task 1.1: explore + deterministic cleaning, writes task01 plots
prepare.py          # decode parquet -> resized images + engineered features (cached .npy)
train.py            # Task 1.2: train classical + CNN, sweep ensemble weight, calibrate, pick winner
predict.py          # Task 1.2: load winner -> artifacts/task02/predictions.csv
train_augmented.py  # Task 1.3: refit on clean+augmented, fine-tune CNN, calibrate at the cap
predict_augmented.py# Task 1.3: load winner -> artifacts/task03/predictions.csv
common.py           # shared: seeds, feature extraction, CNN def, calibration, metrics (ALL scripts import this)
explain.py          # Task 1.4: feature importance, saliency, occlusion, FP/FN analysis
```

Key shared constants in `common.py`: `SEED=0`, `RESIZE=128`, `CNN_RES=64`, `CNN_K=32`,
`MAX_FPR=0.20`, `CALIB_TARGET_FPR=0.17`, `FEATURE_DIM=57`.

---

## 4. The development journey, task by task

### Task 1.1 — Cleaning
Deterministic pipeline: decode → validate → resize to 128px → drop corrupt/degenerate
images. Justified each choice (CPU-friendliness, leakage-safety) rather than applying a
fixed recipe. Class distribution and image-size stats reported in `clean.py` plots.

### Task 1.2 — Modeling under time constraints (the long story)
This is where most of the effort went. The honest progression:

1. **Naive start (Appendix B CNN + classical baseline).** The reference CNN in Appendix B
   has **no BatchNorm**. Out of the box it underperformed badly and trained unstably. The
   PDF even warns Appendix B is "a starting point, not a recommended final solution" — we
   initially under-weighted that warning and lost time.
2. **Breakthrough #1 — BatchNorm in the CNN.** Adding `BatchNorm2d` after each conv was the
   single decisive change. The CNN went from mediocre/unstable to the strongest single model.
3. **Breakthrough #2 — spectral feature engineering (web-researched).** AI/diffusion
   upsamplers leave **periodic high-frequency fingerprints**; real images have smooth
   spectral falloff (Frank et al. 2020; Synthbuster/Bammey 2024; UGAD). We added a
   **residual-spectrum FFT** feature block (FFT of `gray − 3×3 box blur`, log-power, 16
   radial bins + outer-annulus peak/std/kurtosis) plus **per-channel HF stats** (r/g/b outer
   mean + peak). Features grew `32 → 57` dims. All computed on post-resize pixels → leakage-safe.
4. **Breakthrough #3 — calibration target.** We discovered the **calibration→validation FPR
   gap is positive on clean data**: calibrating exactly at 20% breached the gate on
   validation (we saw FPR 0.207 > 0.20 and failed). Fix: calibrate **below** the cap
   (`CALIB_TARGET_FPR=0.17`) to absorb the gap → ended at 0.1915 on validation, safely under.
5. **Ensemble.** A weighted blend `w*classical + (1−w)*cnn`, with `w` swept on the
   calibration split. Winner: `w=0.4` (i.e. 60% CNN). Also added **HistGradientBoosting** as
   a third, stronger classical model.

Result: **0.8077 @ 0.1915** — aim met.

### Task 1.3 — Robustness to augmentation
Augmentation (rescale / JPEG / blur / noise) **destroys the high-frequency spectral signal**
the classical model leans on, so the classical path is fragile here while the CNN (content
features) is robust. Steps:

1. Built a combined clean+augmented feature matrix (58,752 rows), refit the scaler.
2. **Fine-tuned the CNN** from the Task 1.2 checkpoint. First attempt at `lr=5e-4`
   **overfit** — recall peaked at epoch 1 (~0.55) then declined. Fix: **anti-overfit
   schedule** (`AdamW lr=2e-4 + decay to 6e-5`, `weight_decay=1e-4`, 50% augmented batches)
   → recall climbed steadily to ~0.60.
3. **Calibration direction flips.** On augmented data the cal→val gap is **conservative**, so
   calibrating below the cap left recall on the table (deployed FPR was only 0.139). We
   calibrate **at** the cap (`aug_target=0.20`) to use the full budget → **0.6147 @ 0.1765**.
4. Fixed a **threshold-peeking bug**: the per-epoch selection metric used a
   `validation_augmented`-calibrated threshold (optimistic 0.6009); the honest deployed
   number (calibrated on `calibration_augmented`) was 0.5251. Recalibrating on the proper
   split gave the real 0.6147.

### Task 1.4 — Explainability
`explain.py` produces: permutation feature importance (ROC-AUC fallback because
HistGradientBoosting has no `feature_importances_`), CNN saliency montage, occlusion
heatmaps, FP/FN case analysis, and real-vs-AI saliency comparison.
**Headline finding:** spectral-feature importance **drops from 33.8% (Task 1.2) to 24.3%
(Task 1.3)** — concrete evidence that the augmentation-robust model relies *less* on the
fragile spectral shortcut and more on content.

---

## 5. What moved the needle (ranked by impact)

1. **BatchNorm in the CNN** — biggest single jump; turned the CNN into the backbone.
2. **Residual-spectrum + per-channel FFT features** — lifted the classical model and the ensemble.
3. **Calibration-target tuning** — `0.17` clean / `0.20` augmented. Turned gate-*breaching*
   runs into passes (clean) and recovered left-on-table recall (augmented).
4. **Weighted ensemble** (classical + CNN) — captured complementary signals.
5. **HistGradientBoosting** as the classical model — stronger than the linear baseline.
6. **Anti-overfit fine-tune schedule** — the difference between 0.55 and 0.60 on augmented.

---

## 6. What we did wrong (lessons learned)

- **Trusted Appendix B as-is.** No BatchNorm = unstable CNN. Lesson: read the "not
  recommended final solution" warning literally.
- **Calibrated at the cap.** Breached FPR on validation. Lesson: the calibration split is
  not the validation split — leave headroom on clean data.
- **Peeked at the eval split for thresholds.** Optimistic, non-deployable numbers. Lesson:
  *always* calibrate on the calibration split, then report on validation untouched.
- **Trusted wall-clock inside Docker.** The WSL2 VM clock runs 3–17× slow and throttles when
  idle; `sleep`/waits returned early. Lesson: validate by component completion, not timers.
- **sklearn version mismatch** (local vs Docker 1.5.2) made joblib models un-unpicklable
  locally. Lesson: run model-dependent scripts inside the same image that trained them.
- **An iterator idiom hung** (`[None for _ in iter(int,1)]` = infinite loop). Lesson: never
  improvise infinite generators in a batch step.

---

## 7. What we could do better (future improvements)

- Richer CNN (more conv blocks / residual connections) if it still fits the time budget.
- Test-time augmentation or multi-crop for extra robustness on Task 1.3.
- More principled operating point (Youden's J or an explicit cost curve) instead of a fixed
  FPR target.
- Cross-validated calibration to estimate the cal→val gap instead of a hand-tuned offset.
- Higher CNN input resolution (currently 64px) if the budget allows.
- More model families in the ensemble (e.g. a frequency-domain-only specialist).

---

## 8. Environment gotchas (so teammates don't get burned)

- **Docker Desktop crashed twice** — restart via
  `Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"`, then poll `docker ps`.
- **matplotlib** is not in the base image — it's now pinned in `requirements.txt` (needed by
  `clean.py` and `explain.py`).
- **reportlab** is only needed to build the report PDF locally (`report_build.py`); it is
  **not** in `requirements.txt` and must not be (report build is a dev step, not pipeline).
- The big caches (`artifacts/prepared`, `*_images.npy`) make re-runs fast locally but must
  never enter the zip.

---

## 9. Remaining TO-DO before submission (full checklist)

### Blocking
- [ ] **Add team member names to `report.pdf`** (PDF rule, page 1). Currently missing.
      Edit `report.md`, then rebuild: `python report_build.py`.
- [ ] **Set report body to 10pt** (spec: "up to 8 pages (10pt)"). Only 4 pages used, room to spare.
- [ ] **Get a real student ID** for the zip name (`AMLS_Exercise_<student_ID>.zip`).
      (Currently still a placeholder.)

### Clean `solution/` — delete dev-only files before zipping
Remove: `recalibrate.py`, `recalibrate_aug.py`, `recompute_feats.py`, `recompute.log`,
`torch_check.log`, `docker_run.sh`, `docker_run_t3.sh`, `CLEAN_NOTES.md`,
`TASK02_NOTES.md`, `TASK03_NOTES.md`, any `__pycache__/`.

Keep exactly (the runnable pipeline): `Dockerfile`, `common.py`, `clean.py`, `prepare.py`,
`train.py`, `train_augmented.py`, `predict.py`, `predict_augmented.py`, `requirements.txt`,
`explain.py`, `.dockerignore`. **`common.py` is mandatory** (every script imports it).

### Exclude runtime data (page 2: "DO NOT SUBMIT")
- [ ] No `solution/artifacts/` in the zip.
- [ ] No `solution/data/` in the zip.
- [ ] This keeps the zip well under the **20 MB** limit (code-only ≈ <1 MB).

### Final zip layout
```
AMLS_Exercise_<student_ID>.zip   (<= 20 MB)
├── report.pdf
├── explain.py            # Task 1.4 "other-files"
└── solution/             # the 9 pipeline files + .dockerignore listed above
```

### Reproducibility (already done — just verify)
- [x] `CALIB_TARGET_FPR=0.17` (global) reproduces the Task 1.2 winner in `train.py`.
- [x] `aug_target=0.20` reproduces the Task 1.3 winner in `train_augmented.py`.
- [x] Docker image is CPU-only and < 4 GB; runs internet-free.

### Recommended final smoke test
- [ ] Clean end-to-end Docker run of all 6 scripts in order, starting from an **empty**
      `artifacts/`, confirming each finishes within its timeout and both
      `predictions.csv` files are produced in the `row_id,predicted_label` format.
