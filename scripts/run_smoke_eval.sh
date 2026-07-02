#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
mkdir -p results/predictions results/metrics
python -m ssl_asr.evaluate_ctc \
  --model facebook/wav2vec2-base-960h \
  --split validation \
  --subset clean \
  --limit 16 \
  --device "${SSL_ASR_DEVICE:-cuda:1}" \
  --output results/metrics/smoke_wav2vec2.json \
  --predictions results/predictions/smoke_wav2vec2.jsonl

