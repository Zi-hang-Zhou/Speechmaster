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

SpeechMaster-CAR trains a complementarity-aware gain predictor on cached
dev-clean branch outputs and routes utterances with the largest predicted
HuBERT benefit:

| Routed | WER (%) | CER (%) | RTF |
|---:|---:|---:|---:|
| 0% | 2.943 | 0.990 | 0.00128 |
| 10% | 2.516 | 0.817 | 0.00144 |
| 25% | 2.182 | 0.692 | 0.00164 |
| 50% | 2.010 | 0.636 | 0.00197 |
| 75% | 1.887 | 0.603 | 0.00229 |
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
| 10% | 3.205 | 1.005 | 0.00132 |
| 25% | 2.777 | 0.846 | 0.00151 |
| 50% | 2.405 | 0.731 | 0.00181 |
| 75% | 2.315 | 0.694 | 0.00212 |
| 100% | 2.231 | 0.667 | 0.00240 |
| Oracle | 1.920 | 0.658 | - |

## Router Ablation

At 25% dev-clean routing, random routing gives 2.655% WER, duration gives
2.536%, entropy gives 2.314%, the original SpeechMaster peak-entropy router
gives 2.398%, SpeechMaster-CAR gives 2.182%, and the oracle gives 1.450%. At
50% test-clean routing, random gives 2.928%, duration 2.617%, peak-only 2.593%,
peak-entropy 2.602%, SpeechMaster-CAR 2.405%, and oracle 1.920%.

The key conclusion is that risk-aware routing consistently beats random routing,
and a learned complementarity router closes a meaningful fraction of the oracle
gap without rerunning neural inference.

## SpeechMaster-CAR Feature Ablation

CAR uses only fast-branch information at routing time. On test-clean, the full
feature set gives the best 50% budget result.

| Router features | 25% WER (%) | 50% WER (%) |
|---|---:|---:|
| CTC only | 3.031 | 2.635 |
| Duration only | 2.946 | 2.626 |
| CTC + duration | 2.777 | 2.442 |
| Full CAR | 2.777 | 2.405 |

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

## Integrated WavLM Low-Resource Probe

Zihao Long's companion code is integrated under `contrib/wavlm_probe/`. It
trains a 2-layer BiLSTM-CTC character recognizer on frozen WavLM-base+
representations, using 10h LibriSpeech labels. This complements the main
SpeechMaster-CAR system by testing whether continuous SSL states or k-means
discrete units are better as trainable downstream ASR inputs.

Layer sweep on dev-clean, 10h labels:

| WavLM layer | Dev WER (%) | Dev CER (%) |
|---:|---:|---:|
| 1 | 58.40 | 20.04 |
| 4 | 43.77 | 13.49 |
| 6 | 34.32 | 9.90 |
| 8 | 20.75 | 5.89 |
| 10 | 12.57 | 3.49 |
| 12 | 15.66 | 4.30 |

Continuous versus discrete units at layer 10:

| Representation | K | Dev WER (%) | Test WER (%) | Bitrate |
|---|---:|---:|---:|---:|
| Continuous | - | 12.57 | 12.86 | - |
| Discrete | 100 | 36.27 | - | 332 bit/s |
| Discrete | 500 | 23.82 | 23.90 | 448 bit/s |
| Discrete | 1000 | 21.50 | - | 498 bit/s |

The takeaway is consistent with SpeechMaster's unit auditor: discrete SSL units
are compact and usable, but continuous hidden states preserve substantially more
ASR information. This supports treating discrete units as a compression/budget
choice rather than as a free replacement for continuous SSL representations.

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

# Optional teammate companion probe.
cd contrib/wavlm_probe
python src/make_subset.py --root /path/to/LibriSpeech --out data
WAVLM_PROBE_DEVICE=cuda bash src/run_all.sh
```
