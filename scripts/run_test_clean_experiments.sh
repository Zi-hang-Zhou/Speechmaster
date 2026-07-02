#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
DEVICE="${SSL_TEST_DEVICE:-cuda:1}"
LIMIT="${SSL_TEST_LIMIT:-1024}"
mkdir -p results/metrics results/predictions results/tables/test_clean

run_eval() {
  local model="$1"
  local tag="$2"
  python -m ssl_asr.evaluate_ctc \
    --model "$model" \
    --split test \
    --subset clean \
    --limit "$LIMIT" \
    --device "$DEVICE" \
    --output "results/metrics/${tag}.json" \
    --predictions "results/predictions/${tag}.jsonl"
}

run_eval facebook/wav2vec2-base-960h test_clean_wav2vec2_base_final
run_eval facebook/hubert-large-ls960-ft test_clean_hubert_large_final
run_eval facebook/wav2vec2-large-960h-lv60-self test_clean_wav2vec2_large_lv60_final

python -m ssl_asr.speechmaster_router \
  --fast-model facebook/wav2vec2-base-960h \
  --strong-model facebook/hubert-large-ls960-ft \
  --split test \
  --subset clean \
  --limit "$LIMIT" \
  --device "$DEVICE" \
  --route-percentages 0 10 25 50 75 100 \
  --output results/metrics/test_clean_speechmaster_router.json \
  --predictions results/predictions/test_clean_speechmaster_router.jsonl

python -m ssl_asr.route_ablation \
  --predictions results/predictions/test_clean_speechmaster_router.jsonl \
  --output results/metrics/test_clean_route_ablation.json \
  --budgets 0 10 25 50 75 100

python -m ssl_asr.summarize \
  --asr \
    results/metrics/test_clean_wav2vec2_base_final.json \
    results/metrics/test_clean_hubert_large_final.json \
    results/metrics/test_clean_wav2vec2_large_lv60_final.json \
  --speechmaster results/metrics/test_clean_speechmaster_router.json \
  --route-ablation results/metrics/test_clean_route_ablation.json \
  --output-dir results/tables/test_clean
