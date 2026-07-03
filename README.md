# SpeechMaster

SpeechMaster is a reproducible course project for ASR with self-supervised
speech representations. It uses LibriSpeech and pretrained SSL encoders such as
Wav2Vec2 and HuBERT, but the core contribution is not a simple model
comparison. SpeechMaster is a budget-aware framework that merges a fast SSL
recognizer, a stronger SSL recognizer, a complementarity-aware router, and a
discrete-unit budget auditor.

## Research Question

Can a low-resource ASR system improve accuracy without always paying the cost
of the strongest SSL model, while still measuring representation compression?

## Main Contributions

1. SpeechMaster-CAR: Wav2Vec2 decodes every utterance first, then a lightweight
   gain predictor estimates which utterances will benefit from HuBERT under a
   fixed routing budget.
2. A complementarity-aware routing target trained from cached dev-clean branch
   outputs: fast edit count minus strong edit count.
3. A HuBERT discrete-unit budget auditor using k-means over hidden states,
   reporting token rate, bitrate, and codebook effects.
4. A companion WavLM frozen-probe contribution that trains a BiLSTM-CTC head
   and compares continuous hidden states with discrete k-means units under
   low-resource labels.
5. Reproducible WER/CER/RTF experiments, confidence-router baselines, CAR
   feature ablations, layer ablations, unit-codebook ablations, ICASSP-style
   paper, tables, figures, and packaging scripts.

## Directory Layout

```text
configs/          YAML experiment configs
contrib/          teammate WavLM continuous/discrete low-resource probe
data/             downloaded or prepared manifests
paper/            ICASSP 2026 LaTeX paper
results/          metrics, predictions, tables, figures
scripts/          command-line wrappers
src/ssl_asr/      project Python package
```

## Quick Start

```bash
cd /home/zihang/speech
/home/zihang/miniconda3/bin/python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Smoke-test pretrained SSL-ASR evaluation on a small LibriSpeech slice.
bash scripts/run_smoke_eval.sh

# Run the main reproducible evaluation suite, including SpeechMaster-CAR routing.
bash scripts/run_main_experiments.sh

# Run the heavier paper suite used for the submitted tables.
SSL_HEAVY_LIMIT=1024 SSL_HEAVY_UNIT_LIMIT=512 bash scripts/run_heavy_experiments.sh
SSL_TEST_LIMIT=1024 bash scripts/run_test_clean_experiments.sh

# Build the paper PDF if a LaTeX toolchain is installed.
bash scripts/build_paper.sh
```

## Assignment Mapping

- Downstream task: English ASR.
- SSL representations: Wav2Vec2, HuBERT, and WavLM-compatible CTC checkpoints.
- Dataset: LibriSpeech clean validation/test slices by default; low-resource
  fine-tuning configs are provided for train-clean-100 subsets.
- Metrics: WER, CER, RTF, token rate, bitrate, and codebook size.
- Comparisons/ablations: model family, layer fusion, confidence routing,
  SpeechMaster-CAR feature groups, routing budget, codebook size, and unit
  deduplication.

## Current Headline Result

On 1024 LibriSpeech dev-clean utterances:

- Wav2Vec2 fast branch: 2.94% WER.
- SpeechMaster-CAR with 25% routed utterances: 2.18% WER.
- SpeechMaster-CAR with 50% routed utterances: 2.01% WER.
- HuBERT strong branch for all utterances: 1.82% WER.
- Oracle SpeechMaster route: 1.45% WER.

On 1024 LibriSpeech test-clean utterances:

- Wav2Vec2 fast branch: 3.61% WER.
- SpeechMaster-CAR with 25% routed utterances: 2.78% WER.
- SpeechMaster-CAR with 50% routed utterances: 2.40% WER.
- Oracle SpeechMaster route: 1.92% WER.

Companion WavLM low-resource probe from the integrated teammate code:

- Continuous WavLM layer 10 + BiLSTM-CTC, 10h labels: 12.86% test-clean WER.
- Discrete WavLM layer 10 k=500 units: 23.90% test-clean WER at 448 bit/s.
- Layer selection matters: dev WER improves from 58.40% at layer 1 to 12.57%
  at layer 10, then worsens to 15.66% at layer 12.

## Sources Used

- LibriSpeech: https://www.openslr.org/12/
- Hugging Face Wav2Vec2 docs:
  https://huggingface.co/docs/transformers/en/model_doc/wav2vec2
- HuBERT paper: https://arxiv.org/abs/2106.07447
- WavLM paper: https://arxiv.org/abs/2110.13900
- JiWER metrics: https://jitsi.github.io/jiwer/
- ICASSP 2026 Paper Kit:
  https://cmsworkshops.com/ICASSP2026/papers/paper_kit.php
