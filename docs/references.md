# Collected Prerequisite References

This file records how the assignment-provided resources were checked and used
for the final SpeechMaster submission. The selected track is low-resource ASR
with self-supervised speech representations, so ASR resources are used directly
and TTS-only resources are documented as out of scope.

## Assignment Resources

| Resource | Link | Use decision |
|---|---|---|
| LibriSpeech ASR Corpus | https://www.openslr.org/12/ | Main ASR dataset. The official corpus provides 16 kHz read English speech with clean development/test splits and train-clean subsets. |
| LJSpeech Dataset | https://keithito.com/LJ-Speech-Dataset/ | Not used because it is a single-speaker TTS dataset and the project chooses the ASR track. |
| S3PRL | https://github.com/s3prl/s3prl | Used as a methodological reference for SSL representation probing and layer comparison; not imported because the final code is self-contained in PyTorch/Transformers. |
| ESPnet | https://github.com/espnet/espnet | Used as a reference point for end-to-end ASR experiment organization; not imported to keep the submitted system compact and auditable. |
| SpeechBrain | https://github.com/speechbrain/speechbrain | Used as a reference point for speech-toolkit structure; not imported because the project implements the required ASR metrics, routing, and unit analysis directly. |
| Hugging Face Wav2Vec2 | https://huggingface.co/docs/transformers/en/model_doc/wav2vec2 | Used directly for the fast CTC branch and Wav2Vec2 reference systems. |
| Hugging Face HuBERT | https://huggingface.co/docs/transformers/en/model_doc/hubert | Used directly for the strong CTC branch and HuBERT hidden-state unit auditor. |
| Hugging Face WavLM | https://huggingface.co/docs/transformers/en/model_doc/wavlm | Used in the integrated teammate WavLM continuous/discrete low-resource probe. |
| UTMOSv2 | https://github.com/sarulab-speech/UTMOSv2 | Not used because it predicts TTS speech quality/MOS, while the final project evaluates ASR WER/CER. |
| ICASSP 2026 Paper Kit | https://cmsworkshops.com/ICASSP2026/papers/paper_kit.php | Used for the submitted LaTeX paper style. The official `spconf.sty` and `IEEEbib.bst` are included in `paper/`. |

## Project-Specific References

- HuBERT paper: https://arxiv.org/abs/2106.07447
  - Provides the masked prediction and offline cluster motivation behind the
    HuBERT branch and unit-analysis design.
- WavLM paper and official page:
  https://arxiv.org/abs/2110.13900
  https://www.microsoft.com/en-us/research/publication/wavlm-large-scale-self-supervised-pre-training-for-full-stack-speech-processing/
  - Supports the companion WavLM frozen-probe experiment and the layer sweep.
- JiWER: https://jitsi.github.io/jiwer/
  - Provides WER and CER used by the project metrics implementation.

## Scope Decision

The assignment requires choosing either ASR or TTS, not both. SpeechMaster
therefore uses LibriSpeech, Wav2Vec2, HuBERT, WavLM, objective ASR metrics, and
ICASSP formatting directly. LJSpeech and UTMOSv2 are acknowledged but excluded
because adding a TTS branch would dilute the ASR contribution and would not
support the reported WER/CER, routing, or bitrate claims.
