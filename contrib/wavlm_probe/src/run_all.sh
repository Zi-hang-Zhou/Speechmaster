#!/bin/bash
# End-to-end experiment orchestration. Run after features are extracted.
# Usage: bash src/run_all.sh
set -e
cd "$(dirname "$0")/.."

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

EPOCHS="${WAVLM_PROBE_EPOCHS:-25}"
BS="${WAVLM_PROBE_BS:-16}"
DEVICE="${WAVLM_PROBE_DEVICE:-cuda}"

echo "########## Ablation A: layer sweep (continuous, train-10h) ##########"
for L in 1 4 6 8 10 12; do
    python src/train.py \
        --train_manifest data/train-10h.jsonl --dev_manifest data/dev-clean.jsonl \
        --train_feat feats/train-10h --dev_feat feats/dev-clean \
        --layer $L --epochs $EPOCHS --bs $BS \
        --device "$DEVICE" \
        --exp exp/ablA_layers --tag layer$L
done

# pick best layer from ablation A
BEST_LAYER=$(python -c "
import glob,json
best=(1e9,None)
for f in glob.glob('exp/ablA_layers/layer*_result.json'):
    d=json.load(open(f))
    if d['best_dev_wer']<best[0]: best=(d['best_dev_wer'],d['layer'])
print(best[1])
")
echo ">>> best layer = $BEST_LAYER"

echo "########## Ablation B: continuous vs discrete (best layer) ##########"
# continuous baseline at best layer already trained in A (layer$BEST_LAYER)
for K in 100 500 1000; do
    python src/kmeans.py \
        --train_manifest data/train-10h.jsonl --train_feat feats/train-10h \
        --assign_manifests data/train-10h.jsonl data/dev-clean.jsonl data/test-clean.jsonl \
        --assign_feats feats/train-10h feats/dev-clean feats/test-clean \
        --layer $BEST_LAYER --k $K --out_dir exp/km --tag k$K
    python src/train.py \
        --train_manifest data/train-10h.jsonl --dev_manifest data/dev-clean.jsonl \
        --train_feat feats/train-10h --dev_feat feats/dev-clean \
        --layer $BEST_LAYER \
        --discrete_train exp/km/k${K}_train-10h.npz \
        --discrete_dev exp/km/k${K}_dev-clean.npz \
        --discrete_vocab $K --epochs $EPOCHS --bs $BS \
        --device "$DEVICE" \
        --exp exp/ablB_discrete --tag disc_k$K
done

echo "########## Final test-clean eval ##########"
# continuous best layer on test
python src/decode.py \
    --test_manifest data/test-clean.jsonl --test_feat feats/test-clean \
    --ckpt exp/ablA_layers/layer${BEST_LAYER}_best.pt --layer $BEST_LAYER \
    --device "$DEVICE" \
    --out exp/test_continuous.json --dump_hyp exp/hyp_continuous.txt
# discrete k=500 on test (representative)
python src/decode.py \
    --test_manifest data/test-clean.jsonl --test_feat feats/test-clean \
    --ckpt exp/ablB_discrete/disc_k500_best.pt --layer $BEST_LAYER \
    --discrete_test exp/km/k500_test-clean.npz --discrete_vocab 500 \
    --device "$DEVICE" \
    --out exp/test_discrete_k500.json --dump_hyp exp/hyp_discrete_k500.txt

echo "########## Optional Ablation C: data scale 1h vs 10h ##########"
python src/train.py \
    --train_manifest data/train-1h.jsonl --dev_manifest data/dev-clean.jsonl \
    --train_feat feats/train-1h --dev_feat feats/dev-clean \
    --layer $BEST_LAYER --epochs $EPOCHS --bs $BS \
    --device "$DEVICE" \
    --exp exp/ablC_scale --tag scale_1h

python src/summarize.py
echo "ALL_EXPERIMENTS_DONE best_layer=$BEST_LAYER"
