#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
DEVICE="${SSL_ASR_DEVICE:-cuda:1}"
LIMIT="${SSL_ASR_LIMIT:-256}"
mkdir -p results/metrics results/predictions results/tables

python -m ssl_asr.evaluate_ctc --model facebook/wav2vec2-base-960h \
  --split validation --subset clean --limit "$LIMIT" --device "$DEVICE" \
  --output results/metrics/wav2vec2_final.json \
  --predictions results/predictions/wav2vec2_final.jsonl

python -m ssl_asr.evaluate_ctc --model facebook/wav2vec2-base-960h \
  --split validation --subset clean --limit "$LIMIT" --device "$DEVICE" --blend last4 \
  --output results/metrics/wav2vec2_last4.json \
  --predictions results/predictions/wav2vec2_last4.jsonl

python -m ssl_asr.evaluate_ctc --model facebook/hubert-large-ls960-ft \
  --split validation --subset clean --limit "$LIMIT" --device "$DEVICE" \
  --output results/metrics/hubert_final.json \
  --predictions results/predictions/hubert_final.jsonl

python -m ssl_asr.unit_analysis --model facebook/hubert-base-ls960 \
  --split validation --subset clean --limit "${SSL_UNIT_LIMIT:-128}" --device "$DEVICE" \
  --codebooks 50 100 200 500 \
  --output results/metrics/hubert_units.json

python -m ssl_asr.speechmaster_router \
  --fast-model facebook/wav2vec2-base-960h \
  --strong-model facebook/hubert-large-ls960-ft \
  --split validation --subset clean --limit "$LIMIT" --device "$DEVICE" \
  --route-percentages 0 25 50 75 100 \
  --output results/metrics/speechmaster_router.json \
  --predictions results/predictions/speechmaster_router.jsonl

python -m ssl_asr.summarize \
  --asr results/metrics/wav2vec2_final.json results/metrics/wav2vec2_last4.json results/metrics/hubert_final.json \
  --units results/metrics/hubert_units.json \
  --speechmaster results/metrics/speechmaster_router.json \
  --output-dir results/tables
