#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
mkdir -p results/checkpoints results/metrics/finetune

run_train() {
  local tag="$1"
  local limit="$2"
  local steps="$3"
  local extra="${4:-}"
  CUDA_VISIBLE_DEVICES="${SSL_TRAIN_GPU:-3}" python -m ssl_asr.train_ctc \
    --model "${SSL_TRAIN_MODEL:-facebook/wav2vec2-base}" \
    --processor "${SSL_TRAIN_PROCESSOR:-facebook/wav2vec2-base-960h}" \
    --train-limit "$limit" \
    --eval-limit "${SSL_TRAIN_EVAL_LIMIT:-512}" \
    --max-steps "$steps" \
    --batch-size 4 \
    --grad-accum 8 \
    --eval-steps 100 \
    --save-steps 100 \
    --lr "${SSL_TRAIN_LR:-3e-4}" \
    --warmup-steps "${SSL_TRAIN_WARMUP_STEPS:-100}" \
    --fp16 \
    --gradient-checkpointing \
    $extra \
    --output-dir "results/checkpoints/${tag}" \
    --metrics-output "results/metrics/finetune/${tag}.json"
}

run_train wav2vec2_100utt_ssl_feature_frozen 100 400 "--freeze-feature-encoder"
run_train wav2vec2_500utt_ssl_feature_frozen 500 800 "--freeze-feature-encoder"
run_train wav2vec2_1000utt_ssl_feature_frozen 1000 1200 "--freeze-feature-encoder"
run_train wav2vec2_1000utt_ssl_head_only 1000 800 "--freeze-encoder"
