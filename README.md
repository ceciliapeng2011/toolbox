A collection of OpenVINO dev & valiation scripts, for Linux and Windows.

## 1. Accuracy
|  script  |        comment  |
|-----------------:|:-----------|
| run_wwb_grid.sh   | `Run wwb across models, kvcache precision, xattn thresholds, and other ov_configs.` |
| summarize_metrics.sh   | `Collect wwb reports.` |
| filter_per_question_by_similarity.sh   | `Locate wwb test case that fails.` |

## 2. Performance benchmark
|  script  |        comment  |
|-----------------:|:-----------|
| run.benchmark.sh   | `Benchmark across models, kvcache precision, xattn thresholds, and other ov_configs.` |

## 3. Debug & Smoke
|  script  |        comment  |
|-----------------:|:-----------|
| dump4debug.sh   | `Debug with tensor binary/text dumps to PageAttention primitives. Also smoke test.` |

## 4. MISC
|  script  |        comment  |
|-----------------:|:-----------|
| build.ov.sh   | `OpenVINO build` |
| build.genai.sh   | `OpenVINO.GENAI build` |
| download_models.sh   | `Download models from model cache.` |
| prepare_wwb.sh   | `Set up wwb.` |
