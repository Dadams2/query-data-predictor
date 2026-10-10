"""Destination-recovery benchmark: can a scorer anticipate where a trajectory is heading?

A fixed table contains planted destinations of three kinds (structural,
distributional, compositional), plus a shared-context pair in which a
structural and a distributional destination live in the same context, so that
recognising the context does not identify the destination. Each trajectory is a sequence of selection
queries over that table that gradually reveals one destination. After every
step, a copy of each model scores a shared evaluation result; the target is
the extension of the active destination within that result. Probes never enter
model history, models never see labels or future steps, and entity IDs live in
the DataFrame index only.

Targets are generator-owned, not human-interest judgements.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from functools import lru_cache
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .recommender.multidimensional_interestingness_recommender import MultiDimensionalInterestingnessRecommender
from .recommender.frequency_recommender import FrequencyRecommender
from .recommender.similarity_recommender import SimilarityRecommender
from .recommender.clustering_recommender import ClusteringRecommender
from .recommender.random_recommender import RandomRecommender

SCHEMA = {
    'region': ['ASIA', 'EUROPE', 'AMERICA', 'AFRICA', 'MIDDLE_EAST'],
    'category': ['TECHNOLOGY', 'HEALTHCARE', 'AUTOMOTIVE', 'FINANCE', 'RETAIL'],
    'priority': ['LOW', 'STANDARD', 'URGENT'],
    'ship_mode': ['AIR', 'SEA', 'RAIL', 'TRUCK'],
    'channel': ['ONLINE', 'STORE', 'PHONE'],  # never part of a destination
}
NUMERIC = 'price'
COLUMNS = list(SCHEMA) + [NUMERIC]

# Paired destinations of each kind share a region, so early region-level steps
# are consistent with both; the category step is the first distinguishing one.
CONTEXTS = {
    'structural': ({'region': 'ASIA', 'category': 'TECHNOLOGY'},
                   {'region': 'ASIA', 'category': 'HEALTHCARE'}),
    'distributional': ({'region': 'EUROPE', 'category': 'FINANCE'},
                       {'region': 'EUROPE', 'category': 'RETAIL'}),
    'compositional': ({'region': 'AMERICA', 'category': 'AUTOMOTIVE'},
                      {'region': 'AMERICA', 'category': 'TECHNOLOGY'}),
    # Both destinations share one context: 0 is structural (URGENT raised), 1 is
    # distributional (heavy price tail). Only the pattern step distinguishes them.
    'shared': ({'region': 'MIDDLE_EAST', 'category': 'FINANCE'},
               {'region': 'MIDDLE_EAST', 'category': 'FINANCE'}),
}
SHARED_PATTERNS = ('structural', 'distributional')  # pattern of each shared destination
KINDS = tuple(CONTEXTS)
CONDITIONS = ('clear', 'ambiguous', 'detour', 'switch', 'null')
EXPOSURES = ('holdout', 'exposed')
TRAJECTORY_LENGTH = 8
SWITCH_STEP = 4
PRIMARY_K = 10
FINAL_BUDGETS = (5, 10, 25)
SESSION_BUDGET_SECONDS = 300  # per method per session
# Compositional destinations change the ship_mode-priority relationship inside
# the context while leaving the priority marginal approximately unchanged.
COMPOSITION = {'AIR': 'URGENT', 'SEA': 'LOW', 'RAIL': 'STANDARD'}


@dataclass(frozen=True)
class BenchmarkConfig:
    n_rows: int = 12000
    price_log_mean: float = 4.0
    price_log_sigma: float = .35
    structural_confidence: float = .8
    tail_fraction: float = .25
    tail_multiplier: float = 4.
    tail_quantile: float = .99
    compositional_confidence: float = .9
    # Shared context: weaker planting so that the exclusive extensions
    # (urgent but not tail, tail but not urgent) both have enough rows.
    shared_urgent_fraction: float = .45
    shared_tail_fraction: float = .3
    query_limit: int = 200
    panel_extension: int = 15     # per destination
    panel_context_other: int = 10  # per destination: in context, not in extension
    panel_background: int = 50


CONFIG = BenchmarkConfig()

# Original MDI settings, as used in the first submission.
MDI_ORIGINAL_CONFIG = {
    'discretization': {'enabled': True, 'method': 'equal_width', 'bins': 5, 'save_params': False},
    'association_rules': {'min_support': .08, 'min_threshold': .1},
    'summaries': {'desired_size': 10},
    'recommendation': {'mode': 'top_k', 'top_k': 25},
    'multidimensional_interestingness': {
        'alpha': .5, 'beta': .3, 'gamma': .2,
        'rule_decay_rate': .05, 'summary_decay_rate': .1,
        'association_weights': {'confidence': .3, 'support': .2, 'lift': .3, 'j_measure': .2},
        'diversity_weights': {'shannon': .25, 'simpson': .2, 'gini': .2, 'berger': .15, 'mcintosh': .2},
    },
}
# Revised MDI: rank-normalised components, attribute-level diversity and
# equal-frequency bins for numeric columns. The result-delta signal is off in
# the main model and evaluated only as an ablation (mdi_with_delta, delta_only).
MDI_CONFIG = deepcopy(MDI_ORIGINAL_CONFIG)
MDI_CONFIG['discretization']['method'] = 'equal_freq'
MDI_CONFIG['multidimensional_interestingness'].update(
    alpha=.4, beta=.4, gamma=.2, delta_weight=0., delta_decay_rate=.1,
    normalize_components=True, diversity_mode='attribute', narrowing_margin=.05)
METHODS = ('mdi', 'mdi_original', 'association_only', 'mdi_no_novelty', 'mdi_with_delta', 'delta_only', 'mdi_reset',
           'mdi_decay0', 'mdi_decay020', 'competing_models', 'value_profile', 'context_oracle', 'recurrence',
           'predicate_match', 'frequency', 'similarity', 'clustering', 'random')


# --------------------------------------------------------------------------- table

def make_table(seed: int, config: BenchmarkConfig = CONFIG) -> tuple[pd.DataFrame, dict]:
    """Deterministic table plus the data-defined price threshold for tails."""
    rng = np.random.default_rng([seed, 2027])
    n = config.n_rows
    table = pd.DataFrame({c: rng.choice(v, size=n) for c, v in SCHEMA.items()})
    table[NUMERIC] = np.round(rng.lognormal(config.price_log_mean, config.price_log_sigma, n), 2)
    table.index = pd.RangeIndex(n, name='entity_id')

    for kind, contexts in CONTEXTS.items():
        for context in contexts:
            idx = table.index[context_mask(table, context)]
            if kind == 'structural':
                urgent = rng.random(len(idx)) < config.structural_confidence
                table.loc[idx[urgent], 'priority'] = 'URGENT'
            elif kind == 'distributional':
                tail = rng.random(len(idx)) < config.tail_fraction
                table.loc[idx[tail], NUMERIC] = np.round(table.loc[idx[tail], NUMERIC] * config.tail_multiplier, 2)
            elif kind == 'shared':
                if context is contexts[1]:
                    continue  # one context, planted once
                urgent = rng.random(len(idx)) < config.shared_urgent_fraction
                table.loc[idx[urgent], 'priority'] = 'URGENT'
                tail = rng.random(len(idx)) < config.shared_tail_fraction
                table.loc[idx[tail], NUMERIC] = np.round(table.loc[idx[tail], NUMERIC] * config.tail_multiplier, 2)
            else:
                modes = table.loc[idx, 'ship_mode']
                conform = rng.random(len(idx)) < config.compositional_confidence
                for mode, priority in COMPOSITION.items():
                    chosen = idx[(modes == mode).to_numpy() & conform]
                    table.loc[chosen, 'priority'] = priority

    untouched = ~np.logical_or.reduce([context_mask(table, c)
                                       for c in CONTEXTS['distributional'] + CONTEXTS['shared'][:1]])
    threshold = float(table.loc[untouched, NUMERIC].quantile(config.tail_quantile))
    return table, {'tail_threshold': threshold}


def context_mask(frame: pd.DataFrame, context: dict) -> pd.Series:
    return frame[list(context)].eq(pd.Series(context)).all(axis=1)


def extension_mask(frame: pd.DataFrame, kind: str, destination: int, meta: dict) -> pd.Series:
    """Rows that instantiate a destination, defined from observable values only."""
    in_context = context_mask(frame, CONTEXTS[kind][destination])
    if kind == 'shared':
        kind = SHARED_PATTERNS[destination]
    if kind == 'structural':
        return in_context & frame.priority.eq('URGENT')
    if kind == 'distributional':
        return in_context & frame[NUMERIC].gt(meta['tail_threshold'])
    conforming = pd.Series(False, index=frame.index)
    for mode, priority in COMPOSITION.items():
        conforming |= frame.ship_mode.eq(mode) & frame.priority.eq(priority)
    return in_context & conforming


def make_panel(table: pd.DataFrame, meta: dict, kind: str, seed: int,
               config: BenchmarkConfig = CONFIG) -> pd.DataFrame:
    """The shared evaluation result: identical for both destinations of a kind.

    Extension rows are drawn from each destination's exclusive extension and
    in-context rows from neither extension, so the two targets are disjoint
    even when the destinations share a context.
    """
    rng = np.random.default_rng([seed, KINDS.index(kind), 7])
    parts = []
    chosen = pd.Index([], dtype=np.int64)
    either = pd.Series(False, index=table.index)
    exts = [extension_mask(table, kind, d, meta) for d in (0, 1)]
    for destination in (0, 1):
        in_context = context_mask(table, CONTEXTS[kind][destination])
        either |= in_context
        ext, other = exts[destination], exts[1 - destination]
        for mask, count in ((ext & ~other, config.panel_extension),
                            (in_context & ~ext & ~other, config.panel_context_other)):
            pool = table.index[mask]
            pool = pool[~pool.isin(chosen)]
            drawn = rng.choice(pool, size=count, replace=False)
            chosen = chosen.append(pd.Index(drawn))
            parts.append(drawn)
    parts.append(rng.choice(table.index[~either], size=config.panel_background, replace=False))
    ids = rng.permutation(np.concatenate(parts))
    return table.loc[ids]


# ---------------------------------------------------------------------- trajectories

def _shared(kind: str, i: int) -> dict:
    region = CONTEXTS[kind][0]['region']
    return {'region': region} if i == 0 else {'region': region, 'channel': SCHEMA['channel'][(i - 1) % 3]}


def _toward(kind: str, destination: int, meta: dict) -> list[dict]:
    """Distinguish the context, then reveal the pattern, then revisit it."""
    context = dict(CONTEXTS[kind][destination])
    if kind == 'shared':
        kind = SHARED_PATTERNS[destination]
    if kind == 'structural':
        patterns = [{**context, 'priority': 'URGENT'}]
    elif kind == 'distributional':
        patterns = [{**context, NUMERIC: ('>', meta['tail_threshold'])}]
    else:
        patterns = [{**context, 'ship_mode': 'AIR'}, {**context, 'ship_mode': 'SEA'}]
    steps = [context] + patterns
    for i, channel in enumerate(SCHEMA['channel'] * 3):
        base = context if i % 2 == 0 else patterns[0]
        steps.append({**base, 'channel': channel})
    return steps


def make_trajectory(kind: str, active: int, condition: str, meta: dict) -> list[dict]:
    """Queries for one session; ambiguity is the delay before the category step."""
    other = 1 - active
    toward, away = _toward(kind, active, meta), _toward(kind, other, meta)
    if condition == 'clear':
        steps = [_shared(kind, 0)] + toward[:7]
    elif condition == 'ambiguous':
        steps = [_shared(kind, i) for i in range(5)] + toward[:3]
    elif condition == 'detour':
        steps = [_shared(kind, 0)] + toward[:7]
        steps[3], steps[6] = away[0], away[1]
    elif condition == 'switch':
        steps = [_shared(kind, 0)] + away[:SWITCH_STEP - 1] + toward[:TRAJECTORY_LENGTH - SWITCH_STEP]
    elif condition == 'null':
        steps = [_shared(kind, i) for i in range(TRAJECTORY_LENGTH)]
    else:
        raise ValueError(f'Unknown condition {condition}')
    assert len(steps) == TRAJECTORY_LENGTH
    return steps


def distinguishing_step(kind: str, condition: str, meta: dict) -> int | None:
    """First step (1-based) whose query differs between the paired directions.

    For switch it is the first such step after the switch, since earlier steps
    distinguish the pre-switch destination. None if the pair never differs.
    """
    a, b = make_trajectory(kind, 0, condition, meta), make_trajectory(kind, 1, condition, meta)
    start = SWITCH_STEP if condition == 'switch' else 0
    for i in range(start, TRAJECTORY_LENGTH):
        if a[i] != b[i]:
            return i + 1
    return None


def context_ceiling(panel: pd.DataFrame, kind: str, destination: int, meta: dict) -> float:
    """Expected precision of ranking the active context first in random order."""
    in_context = context_mask(panel, CONTEXTS[kind][destination])
    return float(extension_mask(panel, kind, destination, meta)[in_context].mean())


def run_query(table: pd.DataFrame, query: dict, excluded: set, rng: np.random.Generator,
              limit: int = CONFIG.query_limit) -> pd.DataFrame:
    mask = pd.Series(True, index=table.index)
    for column, value in query.items():
        if isinstance(value, tuple):
            op, bound = value
            assert op == '>'
            mask &= table[column].gt(bound)
        else:
            mask &= table[column].eq(value)
    ids = table.index[mask]
    if excluded:
        ids = ids[~ids.isin(list(excluded))]
    if len(ids) > limit:
        ids = rng.choice(ids, size=limit, replace=False)
    else:
        ids = rng.permutation(ids)
    return table.loc[ids]


@dataclass
class Session:
    seed: int
    kind: str
    condition: str
    exposure: str
    target: int
    queries: list
    results: list
    panel: pd.DataFrame
    active: list  # active destination at each step (switch changes it)


def make_session(seed: int, kind: str, condition: str, exposure: str, target: int,
                 config: BenchmarkConfig = CONFIG) -> Session:
    """Paired targets share the table, panel and query RNG stream."""
    if kind not in KINDS or condition not in CONDITIONS or exposure not in EXPOSURES or target not in (0, 1):
        raise ValueError('Unknown session specification')
    table, meta = _cached_table(seed, config)
    panel = make_panel(table, meta, kind, seed, config)
    excluded = set(panel.index) if exposure == 'holdout' else set()
    queries = make_trajectory(kind, target, condition, meta)
    rng = np.random.default_rng([seed, KINDS.index(kind), CONDITIONS.index(condition), 11])
    results = [run_query(table, q, excluded, rng, config.query_limit) for q in queries]
    active = [target] * TRAJECTORY_LENGTH
    if condition == 'switch':
        active = [1 - target] * SWITCH_STEP + [target] * (TRAJECTORY_LENGTH - SWITCH_STEP)
    return Session(seed, kind, condition, exposure, target, queries, results, panel, active)


@lru_cache(maxsize=4)
def _cached_table(seed, config):
    return make_table(seed, config)


# ---------------------------------------------------------------------- models

class Recurrence:
    """Ranks candidates by how often their entity was returned earlier."""

    def __init__(self):
        self.counts = Counter()

    def observe(self, rows, query):
        self.counts.update(rows.index)

    def rank(self, panel, k):
        scores = pd.Series([self.counts[i] for i in panel.index], index=panel.index)
        return panel.loc[scores.sort_values(ascending=False, kind='stable').index[:k]]


class PredicateMatch:
    """Query-aware reference: decayed weight of each predicate the analyst used.

    Unlike the result-only models, this reads query predicates, so it measures
    what query-language agnosticism costs.
    """

    def __init__(self, decay=.9):
        self.decay = decay
        self.weights = defaultdict(float)

    def observe(self, rows, query):
        for key in self.weights:
            self.weights[key] *= self.decay
        for column, value in query.items():
            self.weights[(column, value)] += 1.

    def rank(self, panel, k):
        scores = pd.Series(0., index=panel.index)
        for (column, value), weight in self.weights.items():
            if isinstance(value, tuple):
                scores += weight * panel[column].gt(value[1])
            else:
                scores += weight * panel[column].eq(value)
        return panel.loc[scores.sort_values(ascending=False, kind='stable').index[:k]]


class ContextOracle:
    """Reference that is told the active destination's context.

    It ranks the in-context rows of the evaluation result first, in random
    order, so its precision is the ceiling reachable by recognising the context
    alone. Scoring above it requires recovering the pattern.
    """

    def __init__(self, session: 'Session', seed: int):
        self.session, self.seed, self.step = session, seed, 0

    def observe(self, rows, query):
        self.step += 1

    def rank(self, panel, k):
        active = self.session.active[max(self.step - 1, 0)]
        in_context = context_mask(panel, CONTEXTS[self.session.kind][active]).to_numpy()
        noise = np.random.default_rng([self.seed, self.step]).random(len(panel))
        order = np.lexsort((noise, ~in_context))
        return panel.iloc[order[:k]]


def _discretise(frames: list[pd.DataFrame], extra: pd.DataFrame) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """Price bins at the quintiles of the first observed result, applied to history and candidates.

    Edges come from the first (broadest) result rather than the pooled history:
    equal-frequency bins over the pooled history would make its price
    distribution uniform by construction and hide any concentration.
    """
    edges = np.unique(frames[0][NUMERIC].quantile([.2, .4, .6, .8]).to_numpy())
    binned = lambda f: f[COLUMNS].assign(**{NUMERIC: np.searchsorted(edges, f[NUMERIC].to_numpy(), side='right')})
    return [binned(f) for f in frames], binned(extra)


class ValueProfile:
    """History baseline: score = sum over attributes of the decayed session share of the row's value.

    MDI's distributional component without its concentration weights.
    """

    def __init__(self, decay=.1):
        self.decay = decay
        self.results = []

    def observe(self, rows, query):
        self.results.append(rows)

    def rank(self, panel, k):
        if not self.results:
            return panel.iloc[:k]
        history, candidates = _discretise(self.results, panel)
        n = len(history)
        scores = np.zeros(len(panel))
        for i, frame in enumerate(history):
            w = np.exp(-self.decay * (n - 1 - i))
            for column in COLUMNS:
                share = frame[column].value_counts(normalize=True)
                scores += w * candidates[column].map(share).fillna(0.).to_numpy()
        order = np.argsort(-scores, kind='stable')
        return panel.iloc[order[:k]]


class CompetingModels:
    """Bayesian model selection over hypotheses about which attributes drive exploration.

    Adapted from Monadjemi et al.'s competing models for visual exploration. Each
    hypothesis h is a set of at most two attributes (or none). Under h, the rows
    of the next result follow the decayed history distribution of their values
    on h (Dirichlet-smoothed) and are uniform on the other attributes. The
    posterior over h is updated with the per-row mean log-likelihood of each
    result given the earlier ones (each result counts as one observation).
    Candidates are scored by the posterior-weighted predictive probability.
    Reads results only, like MDI.
    """

    def __init__(self, decay=.9, smoothing=1., max_order=2):
        from itertools import combinations
        self.decay, self.smoothing = decay, smoothing
        self.hypotheses = [()] + [h for r in range(1, max_order + 1) for h in combinations(COLUMNS, r)]
        self.results = []

    def observe(self, rows, query):
        self.results.append(rows)

    def _cardinality(self, column):
        return 5 if column == NUMERIC else len(SCHEMA[column])

    def rank(self, panel, k):
        if not self.results:
            return panel.iloc[:k]
        history, candidates = _discretise(self.results, panel)
        log_post = np.zeros(len(self.hypotheses))
        counts = [Counter() for _ in self.hypotheses]
        total = 0.

        def log_predictive(j, frame):
            h = self.hypotheses[j]
            outside = sum(np.log(self._cardinality(c)) for c in COLUMNS if c not in h)
            if not h:
                return np.full(len(frame), -outside)
            size = np.prod([self._cardinality(c) for c in h])
            keys = list(zip(*(frame[c] for c in h)))
            c = np.array([counts[j][key] for key in keys])
            return np.log((c + self.smoothing) / (total + self.smoothing * size)) - outside

        for frame in history:
            if total > 0:
                for j in range(len(self.hypotheses)):
                    log_post[j] += log_predictive(j, frame).mean()
            for j, h in enumerate(self.hypotheses):
                for key in counts[j]:
                    counts[j][key] *= self.decay
                if h:
                    counts[j].update(zip(*(frame[c] for c in h)))
            total = total * self.decay + len(frame)
        post = np.exp(log_post - log_post.max())
        post /= post.sum()
        scores = sum(post[j] * np.exp(log_predictive(j, candidates)) for j in range(len(self.hypotheses)))
        order = np.argsort(-scores, kind='stable')
        return panel.iloc[order[:k]]


class RepositoryModel:
    """Adapter: observing a result is a recommendation call whose output is discarded."""

    def __init__(self, model, reset=False, factory=None):
        self.model, self.reset, self.factory = model, reset, factory

    def observe(self, rows, query):
        if not self.reset:
            self.model.recommend_tuples(rows.copy(), top_k=PRIMARY_K)

    def rank(self, panel, k):
        probe = self.factory() if self.reset else deepcopy(self.model)
        selected = probe.recommend_tuples(panel.copy(), top_k=k)
        return selected if selected is not None else panel.iloc[:0]


def make_model(method: str, seed: int, session: Session | None = None):
    config = deepcopy(MDI_CONFIG)
    config['random'] = {'random_seed': seed}
    md = config['multidimensional_interestingness']
    if method == 'recurrence':
        return Recurrence()
    if method == 'predicate_match':
        return PredicateMatch()
    if method == 'context_oracle':
        if session is None:
            raise ValueError('The context oracle needs the session')
        return ContextOracle(session, seed)
    if method == 'value_profile':
        return ValueProfile()
    if method == 'competing_models':
        return CompetingModels()
    mdi_methods = ('mdi', 'mdi_original', 'association_only', 'mdi_no_novelty', 'mdi_with_delta', 'delta_only',
                   'mdi_reset', 'mdi_decay0', 'mdi_decay020')
    if method in mdi_methods:
        if method == 'mdi_original':
            config = deepcopy(MDI_ORIGINAL_CONFIG)
            config['random'] = {'random_seed': seed}
            md = config['multidimensional_interestingness']
        if method == 'association_only':
            md.update(alpha=1., beta=0., gamma=0., delta_weight=0.)
        if method == 'mdi_no_novelty':
            md.update(alpha=.5, beta=.5, gamma=0.)
        if method == 'mdi_with_delta':
            md.update(alpha=.3, beta=.3, gamma=.1, delta_weight=.3)
        if method == 'delta_only':
            md.update(alpha=0., beta=0., gamma=0., delta_weight=1.)
        if method == 'mdi_decay0':
            md.update(rule_decay_rate=0., summary_decay_rate=0., delta_decay_rate=0.)
        if method == 'mdi_decay020':
            md.update(rule_decay_rate=.2, summary_decay_rate=.2, delta_decay_rate=.2)
        factory = lambda: MultiDimensionalInterestingnessRecommender(deepcopy(config))
        return RepositoryModel(factory(), reset=method == 'mdi_reset', factory=factory)
    cls = {'frequency': FrequencyRecommender, 'similarity': SimilarityRecommender,
           'clustering': ClusteringRecommender, 'random': RandomRecommender}[method]
    return RepositoryModel(cls(config))


# ---------------------------------------------------------------------- evaluation

def precision_at_budget(selected: pd.DataFrame, panel: pd.DataFrame, extension: set, k: int) -> float:
    """Missing recommendations count as misses; IDs must be unique, unmodified candidates."""
    ids = list(selected.index[:k])
    if len(set(ids)) != len(ids) or not set(ids).issubset(panel.index):
        raise ValueError('Model returned duplicate or non-candidate IDs')
    if ids and not selected.loc[ids, panel.columns].equals(panel.loc[ids]):
        raise ValueError('Model changed candidate values')
    budget = min(k, len(panel))
    return sum(i in extension for i in ids) / budget if budget else 0.


def _alarm(signum, frame):
    raise TimeoutError('Per-call timeout (120 seconds)')


def run_session(spec: tuple, methods=METHODS) -> list[dict]:
    logging.basicConfig(level=logging.ERROR)
    seed, kind, condition, exposure, target = spec
    session = make_session(seed, kind, condition, exposure, target)
    _, meta = _cached_table(seed, CONFIG)
    extensions = [set(session.panel.index[extension_mask(session.panel, kind, d, meta)]) for d in (0, 1)]
    ceilings = [context_ceiling(session.panel, kind, d, meta) for d in (0, 1)]
    q_dist = distinguishing_step(kind, condition, meta)
    records = []
    signal.signal(signal.SIGALRM, _alarm)
    for method in methods:
        model = make_model(method, seed, session)
        failed = None
        method_started = time.time()
        for probe in range(TRAJECTORY_LENGTH + 1):
            # A method that exhausts its per-session budget is stopped; its
            # remaining probes are recorded as errors and count as misses.
            if not failed and time.time() - method_started > SESSION_BUDGET_SECONDS:
                failed = f'TimeBudgetExceeded: over {SESSION_BUDGET_SECONDS}s in this session'
            if probe > 0 and not failed:
                try:
                    np.random.seed(seed * 1000 + probe)
                    signal.alarm(120)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        model.observe(session.results[probe - 1], session.queries[probe - 1])
                except Exception as exc:
                    failed = f'{type(exc).__name__}: {exc}'
                finally:
                    signal.alarm(0)
            active = session.active[probe - 1] if probe > 0 else session.active[0]
            budgets = FINAL_BUDGETS if probe == TRAJECTORY_LENGTH else (PRIMARY_K,)
            for k in budgets:
                tic, error = time.time(), failed
                selected = session.panel.iloc[:0]
                if not failed:
                    try:
                        np.random.seed(seed * 1000 + 100 + probe)
                        signal.alarm(120)
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            selected = model.rank(session.panel, k)
                        precision_at_budget(selected, session.panel, extensions[active], k)
                    except Exception as exc:
                        error = f'{type(exc).__name__}: {exc}'
                        selected = session.panel.iloc[:0]
                    finally:
                        signal.alarm(0)
                records.append(dict(
                    seed=seed, kind=kind, condition=condition, exposure=exposure, target=target,
                    active=active, method=method, probe=probe, k=k, error=error,
                    seconds=time.time() - tic,
                    selected_ids=[int(i) for i in selected.index[:k]],
                    precision=precision_at_budget(selected, session.panel, extensions[active], k),
                    other_precision=precision_at_budget(selected, session.panel, extensions[1 - active], k),
                    chance=len(extensions[active]) / len(session.panel),
                    context_ceiling=ceilings[active], distinguishing_step=q_dist))
    return records


def session_specs(seeds, kinds=KINDS, conditions=CONDITIONS, exposures=EXPOSURES):
    return [(s, k, c, e, t) for s in seeds for k in kinds for c in conditions for e in exposures for t in (0, 1)]


def run(output: Path, seeds, methods=METHODS, kinds=KINDS, conditions=CONDITIONS,
        exposures=EXPOSURES, workers=None):
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    tables = output / 'tables'
    tables.mkdir(exist_ok=True)
    table_hashes = {}
    for seed in seeds:
        table, meta = make_table(seed)
        path = tables / f'table_seed{seed}.csv'
        table.to_csv(path)
        table_hashes[seed] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **meta)

    specs = session_specs(seeds, kinds, conditions, exposures)
    # Each session is checkpointed as it completes, so an interrupted run
    # resumes from the saved sessions instead of starting again.
    sessions_dir = output / 'sessions'
    sessions_dir.mkdir(exist_ok=True)
    session_path = lambda spec: sessions_dir / ('_'.join(map(str, spec)) + '.json')
    todo = [spec for spec in specs if not session_path(spec).exists()]
    print(f'{len(specs) - len(todo)}/{len(specs)} sessions already saved', flush=True)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_session, spec, tuple(methods)): spec for spec in todo}
        for i, future in enumerate(as_completed(futures)):
            spec = futures[future]
            tmp = session_path(spec).with_suffix('.tmp')
            tmp.write_text(json.dumps(future.result()))
            tmp.rename(session_path(spec))
            if (i + 1) % 20 == 0 or i + 1 == len(todo):
                print(f'{len(specs) - len(todo) + i + 1}/{len(specs)} sessions ({time.time() - started:.0f}s)',
                      flush=True)
    raw = [r for spec in specs for r in json.loads(session_path(spec).read_text())]
    (output / 'predictions.json').write_text(json.dumps(raw))
    source = Path(__file__)
    manifest = {
        'protocol': 'destination-benchmark-v1', 'seeds': list(seeds), 'methods': list(methods),
        'kinds': list(kinds), 'conditions': list(conditions), 'exposures': list(exposures),
        'benchmark_config': asdict(CONFIG), 'mdi_config': MDI_CONFIG,
        'mdi_original_config': MDI_ORIGINAL_CONFIG, 'schema': SCHEMA,
        'contexts': CONTEXTS, 'trajectory_length': TRAJECTORY_LENGTH, 'switch_step': SWITCH_STEP,
        'primary_k': PRIMARY_K, 'session_budget_seconds': SESSION_BUDGET_SECONDS, 'final_budgets': list(FINAL_BUDGETS), 'tables': table_hashes,
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'seconds': time.time() - started, 'failures': sum(r['error'] is not None for r in raw),
        'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__,
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    summarize(output, raw)


# ---------------------------------------------------------------------- summaries

def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    radius = float(student_t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))) \
        if len(values) > 1 else 0.
    return mean, radius


def earliness(precisions: list[float], threshold: float) -> int:
    """First probe from which the value stays at or above threshold; L+1 if never."""
    first = TRAJECTORY_LENGTH + 1
    for probe in range(TRAJECTORY_LENGTH, -1, -1):
        if precisions[probe] >= threshold:
            first = probe
        else:
            break
    return first


GROUP = ['kind', 'condition', 'exposure', 'method']


def summarize(output: Path, raw: list[dict]):
    frame = pd.DataFrame(raw)
    # Target directions are paired, not independent replications.
    per_seed = frame.groupby(['seed'] + GROUP + ['probe', 'k'])[['precision', 'other_precision', 'chance']] \
        .mean().reset_index()
    per_seed.to_csv(output / 'per_seed.csv', index=False)

    curves = []
    for keys, group in per_seed.groupby(GROUP + ['probe', 'k']):
        mean, radius = mean_ci(group.precision)
        contrast, contrast_radius = mean_ci(group.precision - group.other_precision)
        curves.append(dict(zip(GROUP + ['probe', 'k'], keys), precision=mean, ci95=radius,
                           target_contrast=contrast, contrast_ci95=contrast_radius,
                           chance=group.chance.mean(), n=len(group)))
    curves = pd.DataFrame(curves)
    curves.to_csv(output / 'curves.csv', index=False)

    # Paired earliness at two levels. A probe counts only if, for BOTH target
    # directions on the same panel, precision exceeds precision on the competing
    # destination and reaches a threshold:
    #   context level: 2x chance (recognising the active context is enough);
    #   pattern level: midway between the context ceiling and 1, which ranking
    #   the active context first in random order cannot reach.
    # History-free rankings and Null can never qualify. Lag is earliness minus
    # the first step that distinguishes the pair (for switch, after the switch).
    primary = frame[frame.k == PRIMARY_K].copy()
    beats = primary.precision > primary.other_precision
    primary['hit'] = (primary.precision >= 2 * primary.chance) & beats
    if 'context_ceiling' in primary:
        primary['pattern_hit'] = (primary.precision >= (primary.context_ceiling + 1) / 2) & beats
    paired = []
    for keys, group in primary.groupby(['seed'] + GROUP):
        row = dict(zip(['seed'] + GROUP, keys))
        q_dist = group.distinguishing_step.iloc[0] if 'distinguishing_step' in group else np.nan
        for level, column in (('', 'hit'), ('pattern_', 'pattern_hit')):
            if column not in group:
                continue
            hits = group.groupby('probe')[column].all().sort_index().astype(float).tolist()
            first = earliness(hits, 1.)
            row[level + 'earliness'] = first
            row[level + 'qualified'] = float(first <= TRAJECTORY_LENGTH)
            row[level + 'lag'] = first - q_dist if first <= TRAJECTORY_LENGTH and pd.notna(q_dist) else np.nan
        row['steps_saved'] = max(0, TRAJECTORY_LENGTH - row['earliness'])
        paired.append(row)
    per_seed_early = pd.DataFrame(paired)
    per_seed_early.to_csv(output / 'earliness_per_seed.csv', index=False)
    early = []
    for keys, group in per_seed_early.groupby(GROUP):
        row = dict(zip(GROUP, keys), n=len(group))
        for column in ('earliness', 'steps_saved', 'pattern_earliness'):
            if column in group:
                row[column], row[column + '_ci95'] = mean_ci(group[column])
        for level in ('', 'pattern_'):
            if level + 'qualified' in group:
                row[level + 'qualified'] = group[level + 'qualified'].mean()
                lags = group[level + 'lag'].dropna()
                row[level + 'lag'] = lags.mean() if len(lags) else np.nan
        early.append(row)
    early = pd.DataFrame(early)
    early.to_csv(output / 'earliness.csv', index=False)

    pairs = []
    final = per_seed[(per_seed.probe == TRAJECTORY_LENGTH) & (per_seed.k == PRIMARY_K)]
    for baseline in ('mdi_original', 'mdi_with_delta', 'association_only', 'mdi_no_novelty', 'mdi_reset',
                     'competing_models', 'value_profile', 'context_oracle', 'recurrence', 'predicate_match',
                     'random'):
        if baseline not in set(final.method):
            continue
        joined = final[final.method == 'mdi'].merge(final[final.method == baseline],
                                                    on=['seed', 'kind', 'condition', 'exposure'],
                                                    suffixes=('_mdi', '_base'))
        for keys, group in joined.groupby(['kind', 'condition', 'exposure']):
            mean, radius = mean_ci(group.precision_mdi - group.precision_base)
            pairs.append(dict(zip(['kind', 'condition', 'exposure'], keys), baseline=baseline,
                              difference=mean, ci95=radius))
    pd.DataFrame(pairs).to_csv(output / 'paired_differences.csv', index=False)

    write_tables(output, curves, early)
    write_figure(output, curves)
    # The manuscript tables need every kind, condition and paper method; a
    # subset run (for a quick check) still gets the full tables and curves above.
    needed = set(PAPER_METHODS) | set(BASELINE_METHODS)
    holdout = curves[curves.exposure == 'holdout']
    if (set(KINDS) <= set(holdout.kind) and set(CONDITIONS) <= set(holdout.condition)
            and needed <= set(holdout.method)):
        write_paper_artifacts(output, curves)
    else:
        print('Subset run: skipped the manuscript tables (paper_table.tex, paper_baselines.tex, '
              'paper_switch.pdf), which need all kinds, conditions and methods.')


NAMES = {'mdi': 'MDI', 'mdi_original': 'Naive MDI', 'association_only': 'Association only',
         'mdi_no_novelty': 'MDI, no novelty', 'competing_models': 'Competing Models',
         'value_profile': 'Value profile', 'context_oracle': 'Context oracle',
         'mdi_with_delta': 'MDI + delta', 'delta_only': 'Delta only', 'mdi_reset': 'MDI, no history',
         'mdi_decay0': 'MDI, no decay', 'mdi_decay020': 'MDI, decay .2',
         'recurrence': 'Tuple recurrence', 'predicate_match': 'Predicate match (query-aware)',
         'frequency': 'Frequency', 'similarity': 'Similarity', 'clustering': 'Clustering', 'random': 'Random'}


def write_tables(output: Path, curves: pd.DataFrame, early: pd.DataFrame, exposure='holdout'):
    final = curves[(curves.probe == TRAJECTORY_LENGTH) & (curves.k == PRIMARY_K) & (curves.exposure == exposure)]
    kinds = [k for k in KINDS if k in set(final.kind)]
    conditions = [c for c in CONDITIONS if c in set(final.condition)]
    methods = [m for m in NAMES if m in set(final.method)]

    def table(values, fmt, caption_cols):
        cols = 'l' + 'r' * len(kinds) * len(caption_cols)
        lines = [r'\begin{tabular}{' + cols + '}', r'\toprule',
                 ' & ' + ' & '.join(rf'\multicolumn{{{len(caption_cols)}}}{{c}}{{{k.capitalize()}}}' for k in kinds)
                 + r' \\',
                 'Method & ' + ' & '.join(c for _ in kinds for c in caption_cols) + r' \\', r'\midrule']
        for method in methods:
            cells = []
            for kind in kinds:
                for condition in conditions_for(caption_cols):
                    cells.append(fmt(values(method, kind, condition)))
            lines.append(NAMES[method] + ' & ' + ' & '.join(cells) + r' \\')
        lines += [r'\bottomrule', r'\end{tabular}']
        return '\n'.join(lines) + '\n'

    short = {'clear': 'Clr', 'ambiguous': 'Amb', 'detour': 'Det', 'switch': 'Sw', 'null': 'Null'}
    conditions_for = lambda cols: [c for c in conditions if short[c] in cols]
    cols = [short[c] for c in conditions]

    def final_value(method, kind, condition):
        row = final[(final.method == method) & (final.kind == kind) & (final.condition == condition)]
        return row.precision.iloc[0] if len(row) else np.nan

    def early_value(method, kind, condition):
        row = early[(early.method == method) & (early.kind == kind) & (early.condition == condition)
                    & (early.exposure == exposure)]
        return row.earliness.iloc[0] if len(row) else np.nan

    (output / 'final_precision_table.tex').write_text(table(final_value, lambda v: f'{v:.2f}', cols))
    (output / 'earliness_table.tex').write_text(table(early_value, lambda v: f'{v:.1f}', cols))


def write_figure(output: Path, curves: pd.DataFrame, exposure='holdout'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = curves[(curves.k == PRIMARY_K) & (curves.exposure == exposure)]
    kinds = [k for k in KINDS if k in set(data.kind)]
    conditions = [c for c in CONDITIONS if c in set(data.condition)]
    shown = [m for m in ('mdi', 'mdi_original', 'association_only', 'mdi_reset', 'recurrence',
                         'predicate_match', 'random') if m in set(data.method)]
    # Okabe-Ito palette (colour-blind safe) plus distinct markers.
    palette = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9', '#000000']
    markers = ['o', 's', '^', 'v', 'D', 'P', 'x']
    fig, axes = plt.subplots(len(kinds), len(conditions), figsize=(2.2 * len(conditions), 1.9 * len(kinds)),
                             sharex=True, sharey=True, squeeze=False)
    for r, kind in enumerate(kinds):
        for c, condition in enumerate(conditions):
            ax = axes[r][c]
            for i, method in enumerate(shown):
                rows = data[(data.kind == kind) & (data.condition == condition) & (data.method == method)] \
                    .sort_values('probe')
                ax.errorbar(rows.probe, rows.precision, yerr=rows.ci95, color=palette[i], marker=markers[i],
                            markersize=3, linewidth=1, capsize=1.5, label=NAMES[method])
            chance = data[(data.kind == kind) & (data.condition == condition)].chance.mean()
            ax.axhline(chance, color='grey', linestyle=':', linewidth=.8)
            ax.axhline(2 * chance, color='grey', linestyle='--', linewidth=.6)
            if condition == 'switch':
                ax.axvline(SWITCH_STEP + .5, color='grey', linewidth=.6)
            if r == 0:
                ax.set_title(condition, fontsize=8)
            if c == 0:
                ax.set_ylabel(f'{kind}\nP@{PRIMARY_K}', fontsize=7)
            if r == len(kinds) - 1:
                ax.set_xlabel('steps observed', fontsize=7)
            ax.tick_params(labelsize=6)
            ax.set_ylim(-.02, 1.02)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(4, len(shown)), fontsize=6, frameon=False)
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(output / 'recovery_curves.pdf', metadata={'CreationDate': None})
    fig.savefig(output / 'recovery_curves.png', dpi=200)
    plt.close(fig)


PAPER_METHODS = ('mdi', 'mdi_original', 'association_only', 'mdi_no_novelty', 'mdi_reset', 'predicate_match')
PAPER_KINDS = ('structural', 'distributional', 'compositional')  # Table 3; shared appears in Table 4
BASELINE_METHODS = ('mdi', 'competing_models', 'value_profile', 'frequency', 'similarity', 'clustering',
                    'recurrence', 'random', 'context_oracle')
REFERENCE_METHODS = ('predicate_match', 'context_oracle')  # not ranked: query-aware or told the context
PAPER_NAMES = {'mdi': 'MDI', 'mdi_original': 'Naive MDI', 'association_only': 'Association only',
               'mdi_no_novelty': 'MDI, no novelty', 'competing_models': 'Competing Models',
               'value_profile': 'Value profile', 'context_oracle': 'Context oracle$^{\\dagger}$',
               'mdi_reset': 'MDI, no history', 'predicate_match': 'Predicate match$^{*}$',
               'recurrence': 'Tuple recurrence', 'random': 'Random', 'frequency': 'Frequency',
               'similarity': 'Similarity', 'clustering': 'Clustering'}


def fmt_precision(value: float) -> str:
    return f'{value:.2f}'.lstrip('0')


def bold_if(text: str, condition: bool) -> str:
    """Best value per column in bold; ties are all bold."""
    return rf'\textbf{{{text}}}' if condition else text


def write_paper_artifacts(output: Path, curves: pd.DataFrame, exposure='holdout'):
    """Compact table and one-row figure sized for the manuscript."""
    final = curves[(curves.probe == TRAJECTORY_LENGTH) & (curves.k == PRIMARY_K) & (curves.exposure == exposure)]
    conditions = ('ambiguous', 'detour', 'switch')
    short = {'ambiguous': 'Amb', 'detour': 'Det', 'switch': 'Sw'}
    kinds = PAPER_KINDS
    lines = [r'\begin{tabular}{@{}l' + 'rrr' * len(kinds) + '@{}}', r'\toprule',
             ' & ' + ' & '.join(rf'\multicolumn{{3}}{{c}}{{{k.capitalize()}}}' for k in kinds) + r' \\',
             ''.join(rf'\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}' for i in range(len(kinds))),
             'Method & ' + ' & '.join(short[c] for _ in kinds for c in conditions) + r' \\', r'\midrule']
    cell = lambda m, k, c: final[(final.method == m) & (final.kind == k) & (final.condition == c)]
    values = {m: [round(float(cell(m, k, c).precision.iloc[0]), 2) for k in kinds for c in conditions]
              for m in PAPER_METHODS}
    max_ci = max(float(cell(m, k, c).ci95.iloc[0]) for m in PAPER_METHODS for k in kinds for c in conditions)
    (output / 'paper_table_max_ci.txt').write_text(f'{max_ci:.3f}\n')
    # References (query-aware or told the context) are excluded from bolding.
    contenders = [m for m in PAPER_METHODS if m not in REFERENCE_METHODS]
    best = [max(values[m][j] for m in contenders) for j in range(len(kinds) * len(conditions))]
    for method in PAPER_METHODS:
        cells = [bold_if(fmt_precision(v), v == b and method in contenders) for v, b in zip(values[method], best)]
        lines.append(PAPER_NAMES[method] + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (output / 'paper_table.tex').write_text('\n'.join(lines) + '\n')
    write_baseline_table(output, curves, early_path=output / 'earliness.csv', exposure=exposure)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = curves[(curves.k == PRIMARY_K) & (curves.exposure == exposure) & (curves.condition == 'switch')]
    shown = ('mdi', 'mdi_original', 'association_only', 'competing_models', 'predicate_match')
    palette = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7']
    markers = ['o', 's', '^', 'v', 'D']
    styles = ['-', '--', '-.', ':', '-']
    fig, axes = plt.subplots(1, len(KINDS), figsize=(3.4, 1.55), sharey=True)
    for ax, kind in zip(axes, KINDS):
        for i, method in enumerate(shown):
            rows = data[(data.kind == kind) & (data.method == method)].sort_values('probe')
            ax.plot(rows.probe, rows.precision, color=palette[i], marker=markers[i], markersize=2.2,
                    linewidth=.9, linestyle=styles[i], label=PAPER_NAMES[method].replace('$^{*}$', ' (query-aware)'))
        ax.axhline(data[data.kind == kind].chance.mean(), color='grey', linestyle=':', linewidth=.6)
        ax.axvline(SWITCH_STEP + .5, color='grey', linewidth=.5)
        ax.set_title(kind, fontsize=6.5, pad=2)
        ax.set_xticks(range(0, TRAJECTORY_LENGTH + 1, 2))
        ax.tick_params(labelsize=5.5, length=2, pad=1)
        ax.set_ylim(-.03, 1.03)
        ax.set_xlabel('steps observed', fontsize=6, labelpad=1)
    axes[0].set_ylabel(f'P@{PRIMARY_K}', fontsize=6, labelpad=1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=5, frameon=False, handlelength=2.2,
               columnspacing=1)
    fig.tight_layout(rect=(0, .2, 1, 1), pad=.2, w_pad=.4)
    fig.savefig(output / 'paper_switch.pdf', metadata={'CreationDate': None})
    plt.close(fig)


def write_baseline_table(output: Path, curves: pd.DataFrame, early_path: Path, exposure='holdout'):
    """Clear trajectories: P@10 per kind, and pattern-level recovery against history baselines.

    Rec.: share of (kind, seed) pairs in which paired pattern-level earliness
    is reached. Lag: mean, over those pairs, of pattern earliness minus the
    distinguishing step. Bold marks every ranked method within the 95%
    interval (over seeds) of the best ranked value, so ties within noise are
    not presented as wins.
    """
    final = curves[(curves.probe == TRAJECTORY_LENGTH) & (curves.k == PRIMARY_K) & (curves.exposure == exposure)
                   & (curves.condition == 'clear')]
    per_seed = pd.read_csv(output / 'earliness_per_seed.csv')
    per_seed = per_seed[(per_seed.exposure == exposure) & (per_seed.condition == 'clear')]
    short = {'structural': 'Str', 'distributional': 'Dist', 'compositional': 'Comp', 'shared': 'Shr'}
    lines = [r'\begin{tabular}{@{}l' + 'r' * (len(KINDS) + 2) + '@{}}', r'\toprule',
             rf' & \multicolumn{{{len(KINDS)}}}{{c}}{{P@10}} & \multicolumn{{2}}{{c}}{{Pattern}} \\',
             rf'\cmidrule(lr){{2-{len(KINDS) + 1}}}\cmidrule(l){{{len(KINDS) + 2}-{len(KINDS) + 3}}}',
             'Method & ' + ' & '.join(short[k] for k in KINDS) + r' & Rec. & Lag \\', r'\midrule']
    methods = [m for m in BASELINE_METHODS + ('predicate_match',) if m in set(final.method)]
    methods = [m for m in methods if m not in REFERENCE_METHODS] + [m for m in methods if m in REFERENCE_METHODS]
    cell = lambda m, k: final[(final.method == m) & (final.kind == k)].iloc[0]
    precision = {m: [round(float(cell(m, k).precision), 2) for k in KINDS] for m in methods}
    recovery, recovery_ci, lag = {}, {}, {}
    for m in methods:
        rows = per_seed[per_seed.method == m]
        rate, radius = mean_ci(rows.groupby('seed').pattern_qualified.mean())
        recovery[m], recovery_ci[m] = round(rate * 100), radius * 100
        lags = rows.pattern_lag.dropna()
        lag[m] = round(float(lags.mean()), 1) if len(lags) else np.nan
    contenders = [m for m in methods if m not in REFERENCE_METHODS]
    best_by_kind = [max(contenders, key=lambda m: precision[m][j]) for j in range(len(KINDS))]
    floor = [precision[b][j] - float(cell(b, KINDS[j]).ci95) for j, b in enumerate(best_by_kind)]
    best_recovery = max(contenders, key=lambda m: recovery[m])
    recovery_floor = recovery[best_recovery] - recovery_ci[best_recovery]
    first_reference = True
    for method in methods:
        ranked = method in contenders
        if not ranked and first_reference:
            lines.append(r'\midrule')
            first_reference = False
        cells = [bold_if(fmt_precision(v), ranked and v >= f - 1e-9) for v, f in zip(precision[method], floor)]
        cells.append(bold_if(f'{recovery[method]}\\%', ranked and recovery[method] >= recovery_floor - 1e-9))
        cells.append('--' if np.isnan(lag[method]) else f'{lag[method]:.1f}')
        lines.append(PAPER_NAMES[method] + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (output / 'paper_baselines.tex').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=Path('artifacts/destination'))
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--kinds', nargs='+', choices=KINDS, default=list(KINDS))
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument('--exposures', nargs='+', choices=EXPOSURES, default=list(EXPOSURES))
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--resummarize', action='store_true',
                        help='Rebuild summaries, tables and figure from an existing predictions.json')
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    if args.resummarize:
        plain, packed = args.output / 'predictions.json', args.output / 'predictions.json.gz'
        if plain.exists():
            raw = json.loads(plain.read_text())
        else:
            import gzip
            with gzip.open(packed, 'rt') as f:
                raw = json.load(f)
        summarize(args.output, raw)
        return
    if os.environ.get('PYTHONHASHSEED') != '0':
        print('WARNING: set PYTHONHASHSEED=0; association-rule ordering (and so tie-breaking) '
              'depends on string hashing, so results are only reproducible with a fixed hash seed.',
              file=sys.stderr)
    if (args.output / 'predictions.json').exists():
        parser.error('Output already contains a run; choose a new output directory.')
    run(args.output, args.seeds, args.methods, args.kinds, args.conditions, args.exposures, args.workers)


if __name__ == '__main__':
    main()
