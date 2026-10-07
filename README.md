# From Next Steps to Destinations: reproduction artifact

Code and results for the vision paper *From Next Steps to Destinations:
Anticipating Analytical Interest in Exploratory Data Analysis* (EDBT 2027).
The branch contains only what is needed to reproduce the paper's results:

- the **destination-recovery benchmark**
  (`src/query_data_predictor/destination_benchmark.py`), which tests whether a
  scorer can anticipate where an exploration trajectory is heading;
- the **MDI** scorer, the history baselines and the standard recommenders it
  is compared against (`src/query_data_predictor/recommender/` and the
  benchmark module);
- the **SkyServer log characterisation** quoted in Section 4
  (`tools/sdss_characterise.py`, reading `data/skyserver_sessions.csv.gz`);
- the **results** reported in the paper (`artifacts/destination/` and
  `artifacts/sdss-characterisation/`).

Targets are planted by the generator. They are not human-interest judgements.

| Paper item | Command | Committed result |
|---|---|---|
| Tables 3 and 4, Figure 2, numbers in Section 5 | benchmark (below) | `artifacts/destination/` |
| SkyServer characterisation, Section 4 | `tools/sdss_characterise.py` | `artifacts/sdss-characterisation/summary.json` |

## Setup

```sh
uv sync --frozen
```

The lock pins the library versions that produced the reported results (Python
3.10, numpy 2.2.5, pandas 2.2.3, scikit-learn 1.6.1, mlxtend 0.23.4, scipy
1.15.2, matplotlib 3.10.1). No database is needed.

## Reproduce the paper

**Tests (about a minute).**

```sh
.venv/bin/python -m pytest tests -q
```

The 43 tests cover:

- determinism and the planted signal;
- disjoint targets in the evaluation result, including for the
  shared-context pair, and the context ceiling;
- distinguishing steps, holdout exclusion, switch labels and null
  construction;
- probes leaving model state unchanged;
- the reference models and history baselines;
- novelty monotonicity, precision validation, and both earliness levels.

**Regenerate Tables 3 and 4 and Figure 2 from the committed predictions (about
a minute).** `artifacts/destination/predictions.json.gz` holds every selection
made in the reported run. To rebuild all summaries from it into a fresh
directory and compare:

```sh
mkdir -p /tmp/destination-check
cp artifacts/destination/predictions.json.gz /tmp/destination-check/
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m query_data_predictor.destination_benchmark \
  --output /tmp/destination-check --resummarize
cmp /tmp/destination-check/paper_table.tex     artifacts/destination/paper_table.tex      # Table 3
cmp /tmp/destination-check/paper_baselines.tex artifacts/destination/paper_baselines.tex  # Table 4
cmp /tmp/destination-check/paper_switch.pdf    artifacts/destination/paper_switch.pdf     # Figure 2
```

**Rerun the benchmark from scratch (about 2.5 hours on 11 cores).**

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
  re-running the same command resumes from them.
- **Time budget:** each method has a 300 s wall-clock budget per session.
  Exceeding it records `TimeBudgetExceeded`, and the remaining probes count as
  misses. On a laptop, keep the machine awake (for example with
  `caffeinate -i` on macOS), because time spent asleep counts against the
  budget. Check that `failures` in `manifest.json` is 0. If it is not, delete
  the affected files in `sessions/` and run the same command again. The
  reported run records 0 failures. 14 sessions that first ran while the machine
  slept were re-run this way.
- **Other options:** output directories are never overwritten.
  `--resummarize` rebuilds all summaries, tables and figures from an existing
  `predictions.json`, or from `predictions.json.gz` if only the compressed file
  is present. Subsets for quick checks: `--kinds`, `--conditions`,
  `--exposures`, `--methods`, `--workers`. A subset run writes the full tables
  and curves but skips the manuscript tables, which need every kind,
  condition and method.

**SkyServer characterisation (Section 4, a few minutes).**

```sh
.venv/bin/python tools/sdss_characterise.py --output /tmp/sdss-check
cmp /tmp/sdss-check/summary.json artifacts/sdss-characterisation/summary.json
```

`data/skyserver_sessions.csv.gz` holds, for each of 462 SkyServer SQL sessions
(160,394 queries) from the Sloan Digital Sky Survey logs, every query's text,
type and result size. The script reports:

- identifier lookups and single-row results (70% of queries);
- query templates (99.7% of queries instantiate one of 104);
- refinement steps. Constant-only changes make up 12.5% of transitions, and
  40% of sessions contain three such steps in a row. Steps that add a
  predicate make up 0.02% of transitions.

## Benchmark design

**Table.** 12,000 rows, with categorical `region`, `category`, `priority`,
`ship_mode` and `channel` and a numeric, lognormal `price`. `channel` is never
part of a destination. Each of the first three kinds has two destinations,
each in its own (region, category) context. The shared-context pair plants two
destinations in one context:

| Kind | Contexts | Planted as | Extension (target rows) |
|---|---|---|---|
| Structural | Asia×Technology, Asia×Healthcare | P(priority=URGENT) raised to ≈.87 in context | context ∧ URGENT |
| Distributional | Europe×Finance, Europe×Retail | 25% of context prices ×4 (heavy tail) | context ∧ price > q.99 of untouched prices |
| Compositional | America×Automotive, America×Technology | ship_mode determines priority (AIR→URGENT, SEA→LOW, RAIL→STANDARD, p=.9); the priority marginal stays roughly uniform | context ∧ conforming (mode, priority) |
| Shared context | Middle East×Finance (both) | 0: 45% of rows set URGENT; 1: 30% of prices ×4 | 0: context ∧ URGENT; 1: context ∧ price > threshold |

In the shared-context pair only the pattern step distinguishes the two
destinations, so recognising the context gives no advantage. This pair was
added after MDI was designed. Extensions are defined from observable values
only.

**Evaluation result.** For each kind, 100 rows: for each of the two
destinations, 15 rows of its exclusive extension and 10 in-context rows that
match neither extension, plus 50 background rows from outside both contexts.
Both target directions share the identical result. Random precision is .15.
The *context ceiling*, the expected precision of ranking the active context
first in random order, is .6 (.3 for the shared pair).

**Trajectories.** Eight selection queries, each capped at 200 rows by seeded
sampling:

- *clear*: region, then the context, then the pattern, then revisits. The
  distinguishing step is 2, or 3 for the shared pair;
- *ambiguous*: five region-level steps, consistent with both destinations,
  before the context step at step 6;
- *detour*: as clear, but steps 4 and 7 go towards the competing destination;
- *switch*: steps 2–4 go towards the competitor and steps 5–8 towards the
  target, so the active destination changes after step 4;
- *null*: region-level steps only. The distinguishing category never appears
  and paired histories are identical, so target contrast must average zero.

Pattern steps are `priority=URGENT` (structural), `price > threshold`
(distributional), and `ship_mode=AIR` then `ship_mode=SEA` (compositional).
For the shared pair they are `priority=URGENT` and `price > threshold`.
Revisits add `channel` filters.

**Exposure.** In *holdout* (reported in the paper), the evaluation result's
entities never appear in any trajectory result, so success requires
transferring evidence to unseen rows. In *exposed*, there is no exclusion, so
containment overlap occurs naturally.

**Probes.** Before any step (probe 0) and after each step (1–8), a deep copy of
each model scores the evaluation result. Probes never enter the model's
history. The primary budget is k=10; the final probe also scores k=5 and 25.

**Earliness.** A probe counts only if, for both target directions, precision@10
exceeds precision on the competing destination and reaches a threshold.

- *Context-level* earliness uses 2× random as the threshold.
- *Pattern-level* earliness uses a threshold midway between the context
  ceiling and 1, which ranking the active context first cannot reach.
- *Lag* is earliness minus the distinguishing step, which is the first step
  at which the paired trajectories' queries differ (for switch, the first
  such step after the switch).

## Methods

- **MDI** (`mdi`), the main model. Rank-normalised association (α=.4),
  attribute-level diversity (β=.4) and novelty (γ=.2, with 1/log(2 + f), so
  that unseen values score highest), using equal-frequency bins (5) for
  numeric columns. See Equations 3–6 in the paper.
- **Naive MDI** (`mdi_original`). The same three kinds of evidence summed on
  their raw scales, with summary-level diversity and equal-width bins.
- **Ablations:**
  - association only;
  - MDI without novelty;
  - MDI with no history (a fresh model at every probe);
  - no decay, and decay .2;
  - *MDI + delta* and *delta only*. The result-delta signal is a decayed
    log-gain of attribute values on steps that narrow that attribute. It is
    excluded from MDI because incidental narrowing, such as `channel` filters,
    distracts it.
- **History baselines** (read results only, like MDI):
  - *Competing Models*, adapted from Monadjemi, Garnett and Ottley (IEEE TVCG
    2021). It performs Bayesian model selection over hypotheses that are
    attribute subsets of size at most two, plus a uniform model, using
    Dirichlet-smoothed decayed value counts (decay .9). Each result counts as
    one observation. Rows are ranked by the posterior predictive;
  - *value profile*: the decayed session share of a row's values. This is
    MDI's distributional component without its concentration weights.

  Both bin price at the quintiles of the first observed result.
- **History-free recommenders:** frequency, similarity, clustering, tuple
  recurrence and random.
- **References** (not ranked):
  - the *context oracle*, which is told the active context;
  - *predicate match*, the only method that reads queries. It scores rows by
    the decayed predicates the analyst used, which measures what being
    query-agnostic costs.

MDI's options are config flags on `MultiDimensionalInterestingnessRecommender`
(`normalize_components`, `diversity_mode`, `delta_weight`, `delta_decay_rate`,
`narrowing_margin`). Their defaults give naive MDI.

## Outputs (`artifacts/destination/`)

- `paper_table.tex`, `paper_baselines.tex`, `paper_switch.pdf`: Tables 3 and 4
  and Figure 2 of the paper. Table 4 bolds every ranked method within the 95%
  interval of the best value. `paper_table_max_ci.txt` is the largest 95%
  half-width in Table 3.
- `final_precision_table.tex`, `earliness_table.tex`, `recovery_curves.pdf/png`:
  full results for all methods and conditions.
- `curves.csv`: precision@k and target contrast at every probe, with
  descriptive 95% t-intervals over seeds.
- `earliness.csv`, `earliness_per_seed.csv`: paired earliness at the context
  and pattern levels (9 if never), the share of seeds that qualify, lag, and
  steps saved.
- `paired_differences.csv`: MDI minus each baseline at the final probe.
- `per_seed.csv`: results averaged over the paired directions within each seed.
- `predictions.json.gz`: every selection of the run, one record per session ×
  method × probe × budget.
- `manifest.json`: configurations, table hashes and environment for the run.

The uncompressed `predictions.json`, per-session checkpoints and generated
tables (about 130 MB) are not committed. They are regenerated by the command
above.

## Limits

Trajectories are scripted selection sequences, not simulated analysts or
natural SQL. Destinations, their strength and their extensions are stipulated.
In this benchmark every planted pattern is also a concentration on a few
attributes, so it cannot yet separate several kinds of interest from a single
proximity model. The baselines are simple adaptations, not tuned systems. No
claim about human interest or user benefit follows from these results.
