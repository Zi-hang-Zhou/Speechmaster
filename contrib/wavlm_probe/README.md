# WavLM Continuous/Discrete Low-Resource Probe

This directory integrates Zihao Long's companion low-resource ASR probe into
the SpeechMaster submission. It is not the main SpeechMaster-CAR route, but it
supports the representation-budget part of the project by training a downstream
CTC recognizer on frozen SSL representations.

## Purpose

The main SpeechMaster system uses pretrained CTC recognizers and studies
budgeted routing between fast and strong SSL-ASR branches. This companion probe
asks a complementary question:

> If a downstream ASR head is trained with limited labels, how do continuous
> WavLM hidden states compare with k-means discrete units?

## Pipeline

1. Build LibriSpeech manifests and speaker-balanced 1h/10h subsets.
2. Extract frozen `microsoft/wavlm-base-plus` hidden states for selected layers.
3. Train a small BiLSTM-CTC head on continuous hidden states.
4. Fit k-means codebooks and train the same head on discrete unit sequences.
5. Report WER/CER, layer sweep, data-scale ablation, token rate, and bitrate.

## Reported Companion Results

These numbers are transcribed from the companion PDF supplied with the group
materials:

- Best continuous WavLM layer: layer 10.
- Continuous L10, 10h labels: 12.57% dev WER, 12.86% test-clean WER.
- Discrete L10, k=500: 23.82% dev WER, 23.90% test-clean WER at 448 bit/s.
- Discrete L10, k=1000: 21.50% dev WER at 498 bit/s.
- Data scale: 1h labels gives 69.95% dev WER; 10h labels gives 12.57%.

## Running

Prepare local LibriSpeech folders, then:

```bash
cd contrib/wavlm_probe
python src/make_subset.py --root /path/to/LibriSpeech --out data

python src/extract.py --manifest data/train-10h.jsonl --out feats/train-10h
python src/extract.py --manifest data/dev-clean.jsonl --out feats/dev-clean
python src/extract.py --manifest data/test-clean.jsonl --out feats/test-clean

WAVLM_PROBE_DEVICE=cuda WAVLM_PROBE_EPOCHS=25 bash src/run_all.sh
```

The scripts are intentionally separate from the main SpeechMaster-CAR pipeline
so that the final paper can keep a clean primary system while still including a
reproducible low-resource continuous/discrete unit probe.
