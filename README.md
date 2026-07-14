# AMLS 2026 — AI Image Detection

Reproducible, **CPU-only, internet-free** ML pipeline that detects AI-generated images
(binary: `real = 0`, `ai_generated = 1`; AI source classes 1–5 are merged into class 1).

- **Task 1.2** (clean): recall_ai **0.808** @ FPR_real **0.192** (aim ≥ 0.80, cap ≤ 20%) ✅
- **Task 1.3** (augmented): recall_ai **0.615** @ FPR_real **0.177** (aim ≥ 0.60, cap ≤ 20%) ✅

---

## 1. Where to put the data

All code lives in `solution/`. Data is **runtime-only** (never committed, never in the
submission zip) and must be placed under `solution/data/` with these subfolders:

```
solution/data/
├── train/                    # labeled shards  (cols: image, source_class)  — clean + prepare
├── calibration/              # labeled         (cols: image, source_class)  — threshold tuning
├── validation/               # labeled         (cols: image, source_class)  — clean gate (Task 1.2)
├── calibration_augmented/    # labeled         — augmented threshold tuning (Task 1.3)
├── validation_augmented/     # labeled         — augmented gate (Task 1.3)
└── predict/                  # UNLABELED       (cols: row_id, image)         — holdout to score
```

All splits are Parquet shards (`*.parquet`). Notes:
- `train/`, `calibration*/`, `validation*/` carry a `source_class` column (0 = real, 1–5 = AI).
- `predict/` carries only `row_id, image` and is **intentionally not pre-cached** by
  `prepare.py` (the grader may swap it at evaluation time).
- The folder `solution/data/` is mounted **read-only** by the grader; all output goes to
  `solution/artifacts/` instead.

---

## 2. How to run (native, fastest to iterate)

CPU-only; no internet needed at runtime.

```bash
cd solution
python -m venv .venv && source .venv/bin/activate 
pip install -r requirements.txt

# Run the 7 scripts IN THIS ORDER:
python clean.py              --timeout_seconds 600
python prepare.py            --timeout_seconds 600
python train.py              --timeout_seconds 1800
python predict.py            --timeout_seconds 600
python train_augmented.py    --timeout_seconds 1800
python predict_augmented.py  --timeout_seconds 600
python explain.py                                      # Task 1.4
```

Outputs land in:
- `solution/artifacts/task02/predictions.csv`  (Task 1.2) — header `row_id,predicted_label`
- `solution/artifacts/task03/predictions.csv`  (Task 1.3) — same format
- `solution/artifacts/task02/model_meta.json` / `task03/model_meta.json` — recall/FPR + chosen threshold

> The order matters: each step reads the previous step's artifacts. `common.py` holds the
> shared constants and helpers and is imported by every script — don't move/rename it.

---

## 3. How to run (Docker — matches the grader exactly)

Use this to reproduce the official numbers; local sklearn/torch versions differ from the
image and **cannot unpickle Docker-trained models**.

```bash
cd solution
docker build -t amls .

# mount data read-only
docker run --rm --cpus 8 \
  -v "$PWD/data:/workspace/solution/data:ro" \
  -v "$PWD/artifacts:/workspace/solution/artifacts" \
  amls python clean.py --timeout_seconds 600
# ...repeat for prepare / train / predict / train_augmented / predict_augmented
```

---

## 4. Repository layout

```
amls_project/
├── README.md                 # this file
├── AMLS_2026_Exercise.pdf    # the assignment
├── report.md / report_build.py / report.pdf   # report draft → PDF (report.pdf IS submitted)
└── solution/                 # the pipeline (this folder is submitted)
    ├── Dockerfile
    ├── requirements.txt
    ├── common.py             # shared constants + features + CNN + calibration 
    ├── clean.py  prepare.py  train.py  predict.py
    ├── train_augmented.py    predict_augmented.py
    └── explain.py            # Task 1.4
```

`data/`, `artifacts/`, and `__pycache__/` are git-ignored (see `.gitignore`).
For the submission zip, **exclude `data/` and `artifacts/`**.
