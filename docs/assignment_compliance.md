# Assignment Compliance Checklist

This document maps the final SpeechMaster submission to the course assignment
requirements. It is included to make the grading surface explicit.

## Chosen Track

- Track: low-resource ASR.
- Downstream task: waveform-to-text transcription on LibriSpeech clean splits.
- Main system name: SpeechMaster.
- Paper title: `SpeechMaster: Complementarity-Aware Routing for Self-Supervised ASR with Discrete Unit Budgeting`.

## Required Items

| Requirement | Covered by |
|---|---|
| Use at least one speech self-supervised model or representation | Wav2Vec2, HuBERT, and WavLM are used. |
| Build an ASR/TTS downstream system | ASR is implemented through CTC decoding and budgeted routed recognition. |
| Use objective metrics | WER, CER, RTF, token rate, bitrate, codebook size, and complementarity counts are reported. |
| Include comparison or ablation | Model family, Wav2Vec2 layer blending, CAR feature groups, routing budgets, routing-score baselines, codebook sizes, HuBERT layers, and WavLM continuous/discrete probe are included. |
| Demonstrate independent design | SpeechMaster-CAR learns expected strong-branch edit-count gain from cached branch disagreement and routes only top-budget utterances. |
| Submit ICASSP-style paper | `paper/main.pdf` is built from the ICASSP 2026 template and is 4 pages. |
| Submit reproducible code and support files | `src/ssl_asr/`, `scripts/`, `configs/`, `contrib/wavlm_probe/`, `results/`, and `README.md` are included in the zip. |

## How the Provided Resources Were Used

| Resource from assignment | Use in final submission |
|---|---|
| LibriSpeech ASR Corpus | Main ASR data source; dev-clean/test-clean are used for reported experiments, and train-clean subsets are supported by the WavLM probe. |
| LJSpeech Dataset | Not used because the selected track is ASR rather than TTS. |
| S3PRL | Used as methodological reference for SSL representation probing; not imported because the implementation is self-contained in PyTorch/Transformers. |
| ESPnet | Used as a reference point for end-to-end speech pipelines; not imported to keep the project minimal and auditable. |
| SpeechBrain | Used as a reference point for speech toolkit design; not imported because the project already implements its needed ASR evaluation path. |
| Hugging Face Wav2Vec2 | Used through pretrained CTC checkpoints and processor/model APIs. |
| Hugging Face HuBERT | Used as the strong branch and as the source of discrete unit analysis. |
| Hugging Face WavLM | Used in the integrated teammate continuous/discrete low-resource probe. |
| UTMOSv2 | Not used because UTMOS is for TTS naturalness/MOS prediction, while the project chooses ASR. |
| ICASSP 2026 Paper Kit | The paper uses the provided `spconf.sty`/`IEEEbib.bst` style and follows the 4-page technical-paper format. |

## Requirement Coverage Details

### Low-resource ASR Design

SpeechMaster uses a fast Wav2Vec2 CTC recognizer for all utterances and a strong
HuBERT CTC recognizer only when CAR predicts a positive edit-count gain under a
routing budget. This turns SSL-ASR from "choose one encoder" into "allocate
recognition compute under a fixed budget".

### Continuous and Discrete Representations

- Continuous representations: pretrained Wav2Vec2/HuBERT/WavLM hidden states
  and CTC logits.
- Discrete units: HuBERT hidden states are clustered with MiniBatchKMeans; the
  WavLM probe trains a downstream BiLSTM-CTC head on k-means unit sequences.
- Compression metrics: token rate, deduplicated token rate, bitrate, and
  codebook size are reported.

### Ablations and Comparisons

- Different SSL models: Wav2Vec2 base, HuBERT large, Wav2Vec2 large LV60, WavLM.
- Different layers/representations: Wav2Vec2 final/last4/midlast, HuBERT layers
  `-8/-4/-1`, WavLM layers `1/4/6/8/10/12`.
- Continuous vs. discrete units: WavLM continuous L10 versus k-means units
  `K=100/500/1000`.
- Codebook size: HuBERT and WavLM unit codebooks report bitrate effects.
- Token deduplication: HuBERT unit auditor reports raw and deduplicated rates.
- Training data scale: WavLM probe reports 1h versus 10h label results.
- Error/complementarity analysis: CAR reports fast-better/strong-better/tied
  counts and insertion/deletion/substitution metrics are stored in JSON outputs.

## Final Submission Artifact

The upload file is `speechmaster_icassp2026_submission.zip`. It contains the
paper PDF, LaTeX source, code, scripts, configs, metrics, predictions, tables,
and the integrated WavLM probe. The original teammate PDF and raw `src(1).zip`
are intentionally not included as final artifacts because their useful parts
have been integrated into the final code and paper.
