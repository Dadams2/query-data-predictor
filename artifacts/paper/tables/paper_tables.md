# Paper Tables

## Table 1: Recommendation performance at gap = 1 on SIMBA drill-down sessions

- Experiment: `SIMBA drill-down benchmark`
- Config: `experiments/configs/simba_drilldown.yml`
- Results: `previous-results/paper/simba-drilldown`
- Analysis CSV: `previous-results/paper/simba-drilldown/analysis/summary/raw/all_sessions_metrics.csv`

| Recommender | Precision | Recall | F1 | Overlap |
| --- | ---: | ---: | ---: | ---: |
| MDI (Ours) | 0.292 | 0.283 | 0.287 | 0.292 |
| Similarity | 0.340 | 0.330 | 0.335 | 0.340 |
| Frequency | 0.308 | 0.299 | 0.303 | 0.308 |
| Random | 0.302 | 0.292 | 0.297 | 0.302 |
| Clustering | 0.292 | 0.283 | 0.287 | 0.000 |
| Sampling | 0.271 | 0.261 | 0.266 | 0.271 |

## Table 2: Recommendation performance at gap = 1 on the adversarial benchmark

- Experiment: `Controlled adversarial benchmark`
- Config: `experiments/configs/benchmark_mdi_vs_baselines.yml`
- Results: `previous-results/paper/adversarial-benchmark`
- Analysis CSV: `previous-results/paper/adversarial-benchmark/analysis/summary/raw/all_sessions_metrics.csv`

| Recommender | Precision | Recall | F1 | Overlap |
| --- | ---: | ---: | ---: | ---: |
| MDI (Ours) | 0.292 | 0.249 | 0.260 | 0.295 |
| Clustering | 0.248 | 0.183 | 0.199 | 0.248 |
| Random | 0.239 | 0.176 | 0.191 | 0.243 |
| Sampling | 0.239 | 0.176 | 0.191 | 0.243 |
| Similarity | 0.208 | 0.145 | 0.160 | 0.212 |
| Frequency | 0.198 | 0.135 | 0.150 | 0.202 |

## Table 3: Recommendation performance at gap = 1 on sampled SkyServer SQL sessions

- Experiment: `Sampled SkyServer SQL benchmark`
- Config: `experiments/configs/sdss_vs_baselines.yml`
- Results: `previous-results/paper/sdss-logs`
- Analysis CSV: `previous-results/paper/sdss-logs/analysis/summary/raw/all_sessions_metrics.csv`

| Recommender | Precision | Recall | F1 | Overlap |
| --- | ---: | ---: | ---: | ---: |
| Clustering | 0.037 | 0.033 | 0.033 | 0.037 |
| Frequency | 0.037 | 0.033 | 0.033 | 0.037 |
| Random | 0.037 | 0.033 | 0.033 | 0.037 |
| Sampling | 0.037 | 0.033 | 0.033 | 0.037 |
| Similarity | 0.019 | 0.018 | 0.018 | 0.019 |
| MDI (Ours) | 0.014 | 0.012 | 0.013 | 0.014 |
