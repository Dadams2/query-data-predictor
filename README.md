# From Next Steps to Destinations: reproduction artifact

Code and results for the vision paper *From Next Steps to Destinations:
Anticipating Analytical Interest in Exploratory Data Analysis* (EDBT 2027).
The branch contains only what is needed to reproduce the paper's results:

- the **destination-recovery benchmark**
  (`src/query_data_predictor/destination_benchmark.py`), which tests whether a
  scorer can anticipate where an exploration trajectory is heading;
- the **MDI** scorer and the standard recommenders it is compared against
  (`src/query_data_predictor/recommender/`);
- the **SkyServer audit** quoted in Section 4 (`tools/audit_sdss.py` with
  snapshots in `data/sdss_snapshots/`);
- the **results** reported in the paper (`artifacts/destination/`).

Targets are planted by the generator. They are not human-interest judgements.

## Setup

```sh
uv sync --frozen
```

The lock pins the library versions that produced the reported results (Python
3.10, numpy 2.2.5, pandas 2.2.3, scikit-learn 1.6.1, mlxtend 0.23.4). No
database is needed.

## Reproduce the paper

**Benchmark (Tables 3 and 4, Figure 2, and all numbers in Section 5).** The run
takes about 75 minutes on 10 cores.

```sh
PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl \
  .venv/bin/python -m query_data_predictor.destination_benchmark \
  --output artifacts/destination-rerun --seeds 100 101 102 103 104 105 106 107 108 109
```

Compare `artifacts/destination-rerun/paper_table.tex`, `paper_baselines.tex` and
`paper_switch.pdf` with the committed files in `artifacts/destination/`.

- **`PYTHONHASHSEED=0` is required** for exact reproduction. Association-rule
  ordering, and so tie-breaking, depends on string hashing. The runner warns if
  it is not set.
- **Seeds:** 100–109 are the reported, held-out seeds. Development seeds 0–9
  were used while designing MDI. Seed 999 is used by tests only.
- **Resuming:** each session is checkpointed in `sessions/` as it finishes, and
  re-running the same command resumes from them. Each method has a 300 s budget
  per session. Exceeding it records `TimeBudgetExceeded`, and the remaining
  probes count as misses. In the reported run no method hit the budget.
- **Other options:** output directories are never overwritten.
  `--resummarize` rebuilds all summaries, tables and figures from an existing
  `predictions.json`. Subsets for quick checks: `--kinds`, `--conditions`,
  `--exposures`, `--methods`, `--workers`.

**SkyServer audit (Section 4, Benchmarks).**

```sh
python3 tools/audit_sdss.py
```

This reports 7 sessions, 608 transitions, 96.5% of transitions sharing no rows
with the next result, and 4 sessions whose median result is a single row. The
snapshots are query-result sequences from SkyServer SQL logs (Sloan Digital Sky
Survey).

**Tests.**

```sh
.venv/bin/python -m pytest tests -q
```

The tests cover determinism, planted signal and disjoint extensions, paired
evaluation results, holdout exclusion, switch labels, null construction, probes
leaving model state unchanged, the reference models, precision validation and
earliness.

## Benchmark design

**Table.** 12,000 rows, with categorical `region`, `category`, `priority`,
`ship_mode` and `channel` and a numeric, lognormal `price`. `channel` is never
part of a destination. Six destinations are planted, two of each kind, each in
its own (region, category) context:

| Kind | Contexts (shared region) | Planted as | Extension (target rows) |
|---|---|---|---|
| Structural | Asia×Technology, Asia×Healthcare | P(priority=URGENT) raised to ≈.87 in context | context ∧ URGENT |
| Distributional | Europe×Finance, Europe×Retail | 25% of context prices ×4 (heavy tail) | context ∧ price > q.99 of untouched prices |
| Compositional | America×Automotive, America×Technology | ship_mode determines priority (AIR→URGENT, SEA→LOW, RAIL→STANDARD, p=.9); the priority marginal stays roughly uniform | context ∧ conforming (mode, priority) |

Extensions are defined from observable values only.

**Evaluation result.** For each kind, 100 rows: for each of the two
destinations, 15 extension rows and 10 in-context non-extension rows, plus 50
background rows from outside both contexts. Both target directions share the
identical result. Chance precision is .15.

**Trajectories.** Eight selection queries, each capped at 200 rows by seeded
sampling:

- *clear*: region, then the context (the distinguishing step 2), then the
  pattern, then revisits;
- *ambiguous*: five region-level steps, consistent with both destinations,
  before the context step at step 6;
- *detour*: as clear, but steps 4 and 7 go towards the competing destination;
- *switch*: steps 2–4 go towards the competitor and steps 5–8 towards the
  target, so the active destination changes after step 4;
- *null*: region-level steps only. The distinguishing category never appears
  and paired histories are identical, so target contrast must average zero.

Pattern steps are `priority=URGENT` (structural), `price > threshold`
(distributional), and `ship_mode=AIR` then `ship_mode=SEA` (compositional).
Revisits add `channel` filters.

**Exposure.** In *holdout* (reported in the paper), the evaluation result's
entities never appear in any trajectory result, so success requires
transferring evidence to unseen rows. In *exposed*, there is no exclusion, so
containment overlap occurs naturally.

**Probes.** Before any step (probe 0) and after each step (1–8), a deep copy of
each model scores the evaluation result. Probes never enter the model's
history. The primary budget is k=10; the final probe also scores k=5 and 25.

## Methods

- **MDI** (`mdi`), the main model. Rank-normalised association (α=.4),
  attribute-level diversity (β=.4) and novelty (γ=.2), with equal-frequency
  bins (5) for numeric columns. See Equations 3–6 in the paper.
- **Naive MDI** (`mdi_original`). The same three kinds of evidence summed on
  their raw scales, with summary-level diversity and equal-width bins.
- **Ablations:**
  - association only;
  - MDI with no history (a fresh model at every probe);
  - no decay, and decay .2;
  - *MDI + delta* and *delta only*. The result-delta signal is a decayed
    log-gain of attribute values on steps that narrow that attribute. It is
    excluded from MDI because incidental narrowing, such as `channel` filters,
    distracts it.
- **References:**
  - *predicate match*, the only method that reads queries. It scores rows by
    the decayed predicates the analyst used, which measures what being
    query-agnostic costs;
  - tuple recurrence;
  - frequency, similarity, clustering and random.

MDI's options are config flags on `MultiDimensionalInterestingnessRecommender`
(`normalize_components`, `diversity_mode`, `delta_weight`, `delta_decay_rate`,
`narrowing_margin`). Their defaults give naive MDI.

## Outputs (`artifacts/destination/`)

- `paper_table.tex`, `paper_baselines.tex`, `paper_switch.pdf`: Tables 3 and 4
  and Figure 2 of the paper.
- `final_precision_table.tex`, `earliness_table.tex`, `recovery_curves.pdf/png`:
  full results for all methods and conditions.
- `curves.csv`: precision@k and target contrast at every probe, with
  descriptive 95% t-intervals over seeds.
- `earliness.csv`: paired earliness, the first probe from which precision@10 is
  at least 2× chance and exceeds precision on the competing destination for both
  target directions (9 if never), and steps saved.
- `paired_differences.csv`: MDI minus each baseline at the final probe.
- `per_seed.csv`: results averaged over the paired directions within each seed.
- `manifest.json`: configurations, table hashes and environment for the run.

The raw `predictions.json`, per-session checkpoints and generated tables
(about 70 MB) are not committed. They are regenerated by the command above.

## Limits

Trajectories are scripted selection sequences, not simulated analysts or
natural SQL. Destinations, their strength and their extensions are stipulated.
No claim about human interest or user benefit follows from these results.
