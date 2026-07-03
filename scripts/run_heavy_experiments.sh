#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-32}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
DEVICE_FAST="${SSL_DEVICE_FAST:-cuda:1}"
DEVICE_STRONG="${SSL_DEVICE_STRONG:-cuda:2}"
LIMIT="${SSL_HEAVY_LIMIT:-1024}"
UNIT_LIMIT="${SSL_HEAVY_UNIT_LIMIT:-512}"
mkdir -p results/metrics results/predictions results/tables results/logs

run_eval() {
  local model="$1"
  local tag="$2"
  local extra="${3:-}"
  python -m ssl_asr.evaluate_ctc \
    --model "$model" \
    --split validation \
    --subset clean \
    --limit "$LIMIT" \
    --device "$DEVICE_FAST" \
    $extra \
    --output "results/metrics/${tag}.json" \
    --predictions "results/predictions/${tag}.jsonl"
}

run_eval facebook/wav2vec2-base-960h heavy_wav2vec2_base_final
run_eval facebook/wav2vec2-base-960h heavy_wav2vec2_base_last4 "--blend last4"
run_eval facebook/wav2vec2-base-960h heavy_wav2vec2_base_midlast "--blend midlast"
run_eval facebook/hubert-large-ls960-ft heavy_hubert_large_final
run_eval facebook/wav2vec2-large-960h-lv60-self heavy_wav2vec2_large_lv60_final

python -m ssl_asr.speechmaster_router \
  --fast-model facebook/wav2vec2-base-960h \
  --strong-model facebook/hubert-large-ls960-ft \
  --split validation \
  --subset clean \
  --limit "$LIMIT" \
  --device "$DEVICE_STRONG" \
  --route-percentages 0 10 25 50 75 100 \
  --output results/metrics/heavy_speechmaster_router.json \
  --predictions results/predictions/heavy_speechmaster_router.jsonl

python -m ssl_asr.route_ablation \
  --predictions results/predictions/heavy_speechmaster_router.jsonl \
  --output results/metrics/heavy_route_ablation.json \
  --budgets 0 10 25 50 75 100

python -m ssl_asr.complementarity_router \
  --train-predictions results/predictions/heavy_speechmaster_router.jsonl \
  --eval-predictions results/predictions/heavy_speechmaster_router.jsonl \
  --output results/metrics/heavy_speechmaster_car.json \
  --budgets 0 10 25 50 75 100

for layer in -8 -4 -1; do
  python -m ssl_asr.unit_analysis \
    --model facebook/hubert-base-ls960 \
    --split validation \
    --subset clean \
    --limit "$UNIT_LIMIT" \
    --layer "$layer" \
    --device "$DEVICE_FAST" \
    --codebooks 50 100 200 500 1000 \
    --output "results/metrics/heavy_hubert_units_layer${layer}.json"
done

python -m ssl_asr.summarize \
  --asr \
    results/metrics/heavy_wav2vec2_base_final.json \
    results/metrics/heavy_wav2vec2_base_last4.json \
    results/metrics/heavy_wav2vec2_base_midlast.json \
    results/metrics/heavy_hubert_large_final.json \
    results/metrics/heavy_wav2vec2_large_lv60_final.json \
  --units \
    results/metrics/heavy_hubert_units_layer-8.json \
    results/metrics/heavy_hubert_units_layer-4.json \
    results/metrics/heavy_hubert_units_layer-1.json \
  --speechmaster results/metrics/heavy_speechmaster_car.json \
  --route-ablation results/metrics/heavy_route_ablation.json \
  --output-dir results/tables/heavy
