# Task 1.2 / AMLS 2026 - CPU-only, internet-free at runtime (Appendix A).
# Build pulls wheels from PyPI + the PyTorch CPU index; the grading *runtime*
# needs no network (data is a read-only mount, artifacts a writable mount).
FROM python:3.11-slim

WORKDIR /workspace/solution

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# CPU-only PyTorch from the official CPU wheel index (no CUDA payload).
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu torch==2.5.1

COPY . /workspace/solution
