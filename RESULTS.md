# SpeechMaster Experiment Results

All results below were produced on July 3, 2026 with
`/home/zihang/miniconda3/envs/torch_gpu` on GPU `cuda:1`.

## ASR Evaluation

LibriSpeech clean validation slice, 128 utterances.

| System | Representation | WER (%) | CER (%) | RTF |
|---|---:|---:|---:|---:|
| facebook/wav2vec2-base-960h | final | 2.474 | 0.945 | 0.003 |
| facebook/wav2vec2-base-960h | last4 mean | 2.858 | 1.026 | 0.003 |
| facebook/hubert-large-ls960-ft | final | 1.578 | 0.493 | 0.005 |

Interpretation:

- HuBERT large fine-tuned ASR gives the best recognition quality on this slice.
- Wav2Vec2 final-layer decoding is stronger than last-four-layer averaging,
  which suggests the CTC projection is most calibrated to the final hidden
  state.
- RTF is far below 1 on the RTX 4090, so all evaluated systems run faster than
  real time on the bounded validation slice.

## SpeechMaster Routing

SpeechMaster always runs the fast Wav2Vec2 branch and routes the lowest
confidence utterances to HuBERT according to a CTC uncertainty score.

| Routed utterances | Routed count | WER (%) | CER (%) | Cascade RTF |
|---:|---:|---:|---:|---:|
| 0% | 0 | 2.474 | 0.945 | 0.002 |
| 25% | 32 | 1.877 | 0.633 | 0.003 |
| 50% | 64 | 1.792 | 0.603 | 0.003 |
| 75% | 96 | 1.706 | 0.553 | 0.003 |
| 100% | 128 | 1.578 | 0.493 | 0.004 |
| Oracle | - | 1.195 | 0.382 | - |

Interpretation:

- Routing only 25% of utterances recovers most of the Wav2Vec2-to-HuBERT gap.
- WER improves monotonically as more uncertain samples are routed.
- The oracle route beats both individual branches, which supports the central
  SpeechMaster claim that SSL model errors are complementary.

## Discrete Unit Analysis

HuBERT base, final hidden layer, LibriSpeech clean validation slice, 64
utterances.

| K | Token/s | Dedup token/s | Bit/s | Dedup bit/s |
|---:|---:|---:|---:|---:|
| 50 | 49.86 | 25.42 | 281.41 | 143.45 |
| 100 | 49.86 | 28.15 | 331.28 | 187.05 |
| 200 | 49.86 | 29.51 | 381.14 | 225.58 |
| 500 | 49.86 | 33.31 | 447.05 | 298.68 |

Interpretation:

- Raw HuBERT frame tokens appear at about 50 Hz.
- Deduplication roughly halves the token rate, producing a compact unit stream.
- Larger codebooks reduce k-means distortion but increase bitrate, exposing the
  expected compression/quality tradeoff.

## Reproduction

```bash
cd /home/zihang/speech
PATH=/home/zihang/miniconda3/envs/torch_gpu/bin:$PATH \
  SSL_ASR_DEVICE=cuda:1 SSL_ASR_LIMIT=128 SSL_UNIT_LIMIT=64 \
  bash scripts/run_main_experiments.sh

bash scripts/build_paper.sh
```
