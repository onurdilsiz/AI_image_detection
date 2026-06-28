#!/bin/bash
# Container entrypoint for the Task 1.3 (augmented) Docker run.
# Copies new code + prepared caches + predict data + the Task 2 checkpoint
# (artifacts/task02) into container-local /work, runs train_augmented.py then
# predict_augmented.py, then copies the task03 outputs back to /host.
set -e
echo "[docker] T3 start $(date -u)"
mkdir -p /work/data /work/artifacts/prepared /work/artifacts/task02 /work/artifacts/task03
cp /host/*.py /host/requirements.txt /work/ 2>/dev/null || true
cp -r /host/data/predict /work/data/predict
cp -f /host/artifacts/task02/* /work/artifacts/task02/ 2>/dev/null || true
echo "[docker] copying prepared caches (~1.7GB)..."
cp /host/artifacts/prepared/* /work/artifacts/prepared/
echo "[docker] caches copied $(date -u)"
cd /work
echo "[docker] === train_augmented.py ==="
python -u train_augmented.py --timeout_seconds ${TRAIN_BUDGET:-1800}
echo "[docker] train_augmented done $(date -u)"
echo "[docker] === predict_augmented.py ==="
python -u predict_augmented.py --timeout_seconds 600
echo "[docker] predict_augmented done $(date -u)"
cp -f /work/artifacts/task03/model_meta.json /host/artifacts/task03/ 2>/dev/null || true
cp -f /work/artifacts/task03/train_report.json /host/artifacts/task03/ 2>/dev/null || true
cp -f /work/artifacts/task03/predictions.csv /host/artifacts/task03/ 2>/dev/null || true
cp -f /work/artifacts/task03/cnn_best.pt /host/artifacts/task03/ 2>/dev/null || true
cp -f /work/artifacts/task03/classical.joblib /host/artifacts/task03/ 2>/dev/null || true
cp -f /work/artifacts/task03/feature_scaler.npz /host/artifacts/task03/ 2>/dev/null || true
echo "[docker] outputs copied back $(date -u)"
echo "[docker] ALL_DONE $(date -u)"
