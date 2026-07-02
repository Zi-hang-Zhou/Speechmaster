# Collected Prerequisite References

- Assignment direction used here: low-resource ASR with self-supervised speech
  representations.
- LibriSpeech corpus: https://www.openslr.org/12/
  - Official description: about 1000 hours of 16 kHz read English speech.
- Hugging Face Wav2Vec2 docs:
  https://huggingface.co/docs/transformers/en/model_doc/wav2vec2
  - Wav2Vec2 accepts raw waveform arrays and CTC checkpoints decode token logits.
- HuBERT paper: https://arxiv.org/abs/2106.07447
  - Uses offline clustering and masked prediction of hidden units.
- WavLM paper and official page:
  https://arxiv.org/abs/2110.13900
  https://www.microsoft.com/en-us/research/publication/wavlm-large-scale-self-supervised-pre-training-for-full-stack-speech-processing/
  - Extends SSL speech pretraining with denoising and sequence modeling changes.
- JiWER: https://jitsi.github.io/jiwer/
  - Provides WER and CER used by the project metrics implementation.
- ICASSP 2026 Paper Kit:
  https://cmsworkshops.com/ICASSP2026/papers/paper_kit.php
  - Maximum 5 pages total; up to 4 pages technical content and an optional
    fifth page for references/acknowledgements/ethics statement.
  - The official 2026 `spconf.sty` and `IEEEbib.bst` are included in `paper/`.

