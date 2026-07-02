# SpeechMaster Experiment Results

Results were produced on July 3, 2026 in
`/home/zihang/miniconda3/envs/torch_gpu` using NVIDIA GPUs.

## Main Dev-Clean Suite

LibriSpeech `validation`/clean, 1024 utterances.

| System | Rep. | WER (%) | CER (%) | RTF |
|---|---:|---:|---:|---:|
| Wav2Vec2 base 960h | final | 2.943 | 0.990 | 0.00146 |
| Wav2Vec2 base 960h | last4 | 3.327 | 1.111 | 0.00151 |
| Wav2Vec2 base 960h | midlast | 3.705 | 1.140 | 0.00149 |
| HuBERT large 960h | final | 1.823 | 0.575 | 0.00256 |
| Wav2Vec2 large LV60 | final | 1.636 | 0.532 | 0.00257 |

SpeechMaster routes low-confidence Wav2Vec2 outputs to HuBERT:

| Routed | WER (%) | CER (%) | RTF |
|---:|---:|---:|---:|
| 0% | 2.943 | 0.990 | 0.00128 |
| 10% | 2.668 | 0.875 | 0.00141 |
| 25% | 2.398 | 0.767 | 0.00161 |
| 50% | 2.044 | 0.654 | 0.00194 |
| 75% | 1.921 | 0.610 | 0.00229 |
| 100% | 1.823 | 0.575 | 0.00262 |
| Oracle | 1.450 | 0.520 | - |

## Test-Clean Verification

LibriSpeech `test`/clean, 1024 utterances.

| System | WER (%) | CER (%) | RTF |
|---|---:|---:|---:|
| Wav2Vec2 base 960h | 3.614 | 1.186 | 0.00142 |
| HuBERT large 960h | 2.231 | 0.667 | 0.00249 |
| Wav2Vec2 large LV60 | 1.849 | 0.598 | 0.00248 |

| Routed | WER (%) | CER (%) | RTF |
|---:|---:|---:|---:|
| 0% | 3.614 | 1.186 | 0.00117 |
| 10% | 3.398 | 1.093 | 0.00128 |
| 25% | 3.035 | 0.946 | 0.00146 |
| 50% | 2.602 | 0.797 | 0.00178 |
| 75% | 2.292 | 0.698 | 0.00211 |
| 100% | 2.231 | 0.667 | 0.00240 |
| Oracle | 1.920 | 0.658 | - |

## Router Ablation

At 25% dev-clean routing, random routing gives 2.655% WER, duration gives
2.536%, entropy gives 2.314%, SpeechMaster peak-entropy gives 2.398%, and the
oracle gives 1.450%. At 50% test-clean routing, random gives 2.928%, duration
2.617%, peak-only 2.593%, peak-entropy 2.602%, and oracle 1.920%.

The key conclusion is that risk-aware routing consistently beats random routing,
while the oracle gap shows useful headroom for a learned router.

## Discrete Unit Analysis

HuBERT base units, LibriSpeech dev-clean, 512 utterances. Raw token rate is
about 49.9 tokens/s across layers. Deduplication roughly halves the rate.

| Layer | K | Dedup token/s | Dedup bit/s |
|---:|---:|---:|---:|
| -8 | 100 | 28.08 | 186.58 |
| -8 | 1000 | 35.17 | 350.46 |
| -4 | 100 | 27.14 | 180.34 |
| -4 | 1000 | 29.74 | 296.36 |
| -1 | 100 | 26.67 | 177.19 |
| -1 | 1000 | 34.50 | 343.85 |

Full CSV/LaTeX tables and plots are in `results/tables/heavy/` and
`results/tables/test_clean/`.

## Reproduction

```bash
cd /home/zihang/speech
PATH=/home/zihang/miniconda3/envs/torch_gpu/bin:$PATH \
  SSL_HEAVY_LIMIT=1024 SSL_HEAVY_UNIT_LIMIT=512 \
  bash scripts/run_heavy_experiments.sh

PATH=/home/zihang/miniconda3/envs/torch_gpu/bin:$PATH \
  SSL_TEST_LIMIT=1024 \
  bash scripts/run_test_clean_experiments.sh

bash scripts/build_paper.sh
```
