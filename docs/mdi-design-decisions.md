# MDI design decisions (EDBT 2027 revision)

Why the MDI settings used by `destination_benchmark.py` (`MDI_CONFIG`) differ
from the original model (`MDI_ORIGINAL_CONFIG`, "naive MDI"), and how much
evidence each decision has. The reported results are on the
`edbt2027-submission` tag.

## Process

1. The original model was run on development seeds 0–9 (destination
   benchmark v1: structural, distributional and compositional destinations).
2. Those results were used to diagnose it, and the revision was designed using
   one development seed. Which seed was not recorded.
3. The revised model was run once on held-out seeds 100–109, which the paper
   reports. The paper says the settings were "set on development seeds".
4. v2 added the shared-context pair and fixed novelty (below). Nothing was
   retuned.

The development run is not in the repository or the artifact. It existed
only as untracked files (`artifacts/destination-dev/`, about 27 MB, mostly
`predictions.json`) in the vision-paper checkout. The figures below come from
its `per_seed.csv`. It is regenerable by running the benchmark with
`MDI_ORIGINAL_CONFIG` on `--seeds 0 1 2 3 4 5 6 7 8 9` at the v1 code.

## Decisions

| Setting | Original | Revised | Reason | Evidence |
|---|---|---|---|---|
| Numeric bins | equal-width, 5 | equal-frequency, 5 | Naive MDI could not find distributional destinations. The likely mechanism is rule support, not tail visibility; see below. | Strong, but not isolated; see below. |
| Component scaling | raw sums | rank-normalised per scoring call | Association scores are on a larger scale and dominated the sum. | Diagnosis only; no ablation varies it alone. |
| Diversity | over pattern summaries | per attribute (`diversity_mode: attribute`) | The summariser produced fully generalised summaries that matched every row, so the summary component was constant. On dev seeds, MDI and association only scored identically. | Direct (identical dev scores). |
| α/β/γ | .5/.3/.2 | .4/.4/.2 | Rebalanced once diversity carried signal. | None beyond the dev seed; not tuned. |
| Result-delta (`delta_weight`) | absent | 0 (ablation only) | Incidental narrowing, such as `channel` filters, distracts it. | Reported as ablations `mdi_with_delta` and `delta_only`. |
| Novelty | 1/log(1 + f), 1.0 if unseen | 1/log(2 + f) | The old form ranked a value seen once (1.44) above an unseen one (1.0). | Correctness fix (v2), not tuning. |
| Bins (5), min support (.08), min confidence (.1), decay rates | inherited | unchanged | Kept from the original model. | Not tested, apart from rule decay 0, .05 and .2 (reported ablations). |

## Binning: what the evidence does and does not show

Precision@10 after eight steps, distributional destinations, holdout exposure,
all conditions except null (random is .15):

| Method | Equal-width bins | Equal-frequency bins |
|---|---|---|
| Association only | .069 (dev seeds, v1) | .751 (held-out seeds, v2) |
| Full MDI | .069 (dev seeds, original) / .067 (held-out, naive MDI) | .791 (held-out, revised) |

Association only has a single component, so the scaling and weight changes
cannot explain its jump; the bins can. This is an inference across different
seeds and code versions, not an ablation: no run varies binning with
everything else fixed.

Why the bins matter is not what it first appears. Fitting both methods on a
200-row EUROPE sample (the benchmark's first-result shape), the tail
threshold is about 125:

| Seed | Equal-width (edges, share per bin) | Equal-frequency |
|---|---|---|
| 0 | 16, 70, 123, 177, 231, 285; shares .67, .24, .02, .02, .04 | top bin is price > 80, 20% of rows |
| 100 | 22, 92, 161, 230, 300, 369; shares .81, .10, .06, .02, .00 | top bin is price > 87, 20% of rows |

Equal-width bins do not hide the tail: they separate it more finely than
equal-frequency bins, which lump it with every price above the 80th
percentile. The likely mechanism is the minimum support of .08. Each
equal-width tail bin holds 2–6% of rows, too few to enter any frequent
itemset, so no association rule mentions the tail. The equal-frequency top
bin holds 20% and does. This is a hypothesis; it has not been tested.

The paper (Section 5) attributes naive MDI's failure to raw-scale
association scores and to equal-width bins "hiding" the heavy tail. The
evidence supports binning as the cause, but through rule support rather than
visibility. Raising the bin count or lowering min support may matter as much
as the binning method.

## Caveats for future work

- **Bin edges come from the first result.** `Discretizer` fits edges on the
  first result the model sees (a 200-row, region-level query in the
  benchmark), caches them per column and reuses them. Values above the
  first result's maximum fall into an extra "above range" bin. Whether MDI can
  separate a `price > q.99` destination therefore depends on what the first
  query returned as well as on the bin count. For out-of-result search,
  where candidates come from the database, fit edges on the table's column
  distribution instead.
- **K-means binning is inconsistent.** On the first call it labels rows by
  cluster; on later calls it digitises against the cluster centres as if they
  were edges. Nothing uses it.
- **A binning ablation would settle it:** equal-width vs equal-frequency at
  3, 5 and 10 bins, crossed with min support .02 and .08, on development
  seeds, distributional and shared kinds, with everything else at the
  revised settings.
