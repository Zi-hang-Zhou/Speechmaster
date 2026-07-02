#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" python -m ssl_asr.train_ctc \
  --model facebook/wav2vec2-base \
  --processor facebook/wav2vec2-base-960h \
  --train-limit "${SSL_TRAIN_LIMIT:-1000}" \
  --eval-limit "${SSL_EVAL_LIMIT:-256}" \
  --max-steps "${SSL_MAX_STEPS:-1000}" \
  --batch-size 4 \
  --grad-accum 8 \
  --fp16 \
  --gradient-checkpointing \
  --freeze-feature-encoder \
  --output-dir results/checkpoints/wav2vec2_1k_utt

