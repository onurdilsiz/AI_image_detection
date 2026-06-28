#!/bin/bash
# Container entrypoint for the Task 1.2 Docker validation run.
# Copies the new code + prepared caches + predict data from the bind-mounted
# /host into container-local /work (bind-mount random reads are cripplingly slow
# on Windows/WSL2), trains, predicts, then copies outputs back to /host.
set -e
echo "[docker] start $(date -u)"
mkdir -p /work/data /work/artifacts/prepared /work/artifacts/task02
cp /host/*.py /host/requirements.txt /work/ 2>/dev/null || true
cp -r /host/data/predict /work/data/predict
echo "[docker] copying prepared caches (~1.7GB)..."
cp /host/artifacts/prepared/* /work/artifacts/prepared/
echo "[docker] caches copied $(date -u)"
cd /work
echo "[docker] === train.py ==="
python -u train.py --timeout_seconds ${TRAIN_BUDGET:-1800}
echo "[docker] train done $(date -u)"
echo "[docker] === predict.py ==="
python -u predict.py --timeout_seconds 600
echo "[docker] predict done $(date -u)"
cp -f /work/artifacts/task02/model_meta.json /host/artifacts/task02/ || true
cp -f /work/artifacts/task02/train_report.json /host/artifacts/task02/ || true
cp -f /work/artifacts/task02/predictions.csv /host/artifacts/task02/ || true
cp -f /work/artifacts/task02/cnn_best.pt /host/artifacts/task02/ 2>/dev/null || true
cp -f /work/artifacts/task02/classical.joblib /host/artifacts/task02/ 2>/dev/null || true
cp -f /work/artifacts/task02/feature_scaler.npz /host/artifacts/task02/ 2>/dev/null || true
echo "[docker] outputs copied back $(date -u)"
echo "[docker] ALL_DONE $(date -u)"
