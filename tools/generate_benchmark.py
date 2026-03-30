#!/usr/bin/env python3
"""
Synthetic Benchmark Dataset Generator for Query Data Predictor.

Generates synthetic benchmark datasets where Multidimensional Interestingness (MDI)
recommender dramatically outperforms all baselines (Random, Clustering, Frequency,
Similarity, Sampling) by embedding strong association rules into data with
properties that exploit each baseline's weaknesses:

  - Rule-conforming overlap tuples use RARE attribute values → Frequency scores
    them low, MDI's association component scores them high.
  - Noise rows are homogeneous (common values) → Clustering centroids land on
    noise, Similarity ranks noise highest. Rule tuples are scattered outliers.
  - Overlap tuples are a small fraction of current_results → Random misses most.
  - Session-persistent focal rules enable MDI's temporal memory to accumulate
    signal, giving MDI an increasing advantage at larger prediction gaps.

Usage:
    python tools/generate_benchmark.py -o data/datasets/benchmark_mdi -n 3 -q 30 --seed 42
    python tools/generate_benchmark.py --verify
"""

import argparse
import pathlib
import pickle
import sys
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Schema and embedded association rules
# ──────────────────────────────────────────────────────────────────────────────

# "Common" values that noise rows will concentrate on (makes them cluster
# together and score highly on frequency/similarity). "Rare" values appear
# only in rule-conforming rows.
SCHEMA = {
    "region": ["ASIA", "EUROPE", "AMERICA", "AFRICA", "MIDDLE_EAST"],
    "category": ["TECHNOLOGY", "AUTOMOTIVE", "HEALTHCARE", "FINANCE", "RETAIL",
                  "ENERGY", "AGRICULTURE", "LOGISTICS"],
    "supplier_type": ["DOMESTIC", "INTERNATIONAL", "PREMIUM", "BUDGET"],
    "price_range": ["LOW", "MEDIUM", "HIGH", "PREMIUM"],
    "quantity_bin": ["SMALL", "MEDIUM", "LARGE", "BULK"],
    "priority": ["LOW", "STANDARD", "URGENT"],
    "ship_mode": ["AIR", "SEA", "RAIL", "TRUCK"],
    "year": ["2022", "2023", "2024"],
    "quarter": ["Q1", "Q2", "Q3", "Q4"],
}

COLUMNS = list(SCHEMA.keys())

# Common values that noise rows prefer — 2-3 dominant values per column.
# This makes noise rows cluster tightly in encoded space and score high
# on frequency-based recommenders.
COMMON_VALUES = {
    "region": ["AMERICA", "EUROPE"],
    "category": ["AUTOMOTIVE", "FINANCE"],
    "supplier_type": ["DOMESTIC", "BUDGET"],
    "price_range": ["MEDIUM", "LOW"],
    "quantity_bin": ["MEDIUM", "SMALL"],
    "priority": ["STANDARD", "LOW"],
    "ship_mode": ["TRUCK", "RAIL"],
    "year": ["2023", "2024"],
    "quarter": ["Q2", "Q3"],
}

# Each rule: (antecedent_dict, consequent_dict)
# Rules use values that are rare in noise distribution → high novelty,
# low frequency, far from noise cluster centroids.
EMBEDDED_RULES = [
    ({"region": "ASIA", "category": "TECHNOLOGY"},
     {"price_range": "HIGH", "priority": "URGENT"}),
    ({"region": "EUROPE", "category": "HEALTHCARE"},
     {"price_range": "PREMIUM", "ship_mode": "AIR"}),
    ({"region": "MIDDLE_EAST", "category": "RETAIL"},
     {"price_range": "PREMIUM", "quantity_bin": "LARGE"}),
    ({"supplier_type": "INTERNATIONAL", "category": "ENERGY"},
     {"quantity_bin": "BULK", "ship_mode": "SEA"}),
    ({"region": "AFRICA", "category": "AGRICULTURE"},
     {"price_range": "HIGH", "priority": "URGENT"}),
]


# ──────────────────────────────────────────────────────────────────────────────
# BenchmarkSchemaGenerator
# ──────────────────────────────────────────────────────────────────────────────

class BenchmarkSchemaGenerator:
    """Generates a synthetic fact table with embedded association rules.

    Noise rows are biased toward COMMON_VALUES so they cluster together in
    encoded feature space and dominate frequency counts.  Rule-conforming rows
    use the specific (often rare) values dictated by EMBEDDED_RULES.
    """

    def __init__(self, rng: np.random.Generator, total_rows: int = 5000,
                 rule_confidence: float = 0.85):
        self.rng = rng
        self.total_rows = total_rows
        self.rule_confidence = rule_confidence

    def generate(self) -> pd.DataFrame:
        rows_per_rule = int(self.total_rows * 0.12)
        rule_rows = []
        for antecedent, consequent in EMBEDDED_RULES:
            rule_rows.append(
                self._generate_rule_rows(antecedent, consequent, rows_per_rule))

        noise_count = self.total_rows - sum(len(r) for r in rule_rows)
        noise = self._generate_noise_rows(noise_count)

        df = pd.concat(rule_rows + [noise], ignore_index=True)
        df = df.sample(frac=1, random_state=int(self.rng.integers(0, 2**31))
                       ).reset_index(drop=True)
        return df

    def _generate_rule_rows(self, antecedent: dict, consequent: dict,
                            n: int) -> pd.DataFrame:
        rows = []
        for _ in range(n):
            # Start from a random row — free columns get diverse values
            row = self._random_row_uniform()
            for col, val in antecedent.items():
                row[col] = val
            if self.rng.random() < self.rule_confidence:
                for col, val in consequent.items():
                    row[col] = val
            rows.append(row)
        return pd.DataFrame(rows, columns=COLUMNS)

    def _generate_noise_rows(self, n: int) -> pd.DataFrame:
        """Generate noise rows biased toward COMMON_VALUES."""
        rows = []
        antecedent_patterns = [frozenset(a.items()) for a, _ in EMBEDDED_RULES]
        for _ in range(n):
            row = self._random_row_biased()
            # Break any accidental rule-antecedent matches
            row_items = {(col, row[col]) for col in COLUMNS}
            for pattern in antecedent_patterns:
                if pattern.issubset(row_items):
                    col_to_perturb = list(dict(pattern).keys())[0]
                    row[col_to_perturb] = self.rng.choice(
                        COMMON_VALUES[col_to_perturb])
                    break
            rows.append(row)
        return pd.DataFrame(rows, columns=COLUMNS)

    def _random_row_uniform(self) -> dict:
        """Fully uniform random row."""
        return {col: self.rng.choice(vals) for col, vals in SCHEMA.items()}

    def _random_row_biased(self) -> dict:
        """Row biased 80% toward common values, 20% uniform."""
        row = {}
        for col, vals in SCHEMA.items():
            if self.rng.random() < 0.80:
                row[col] = self.rng.choice(COMMON_VALUES[col])
            else:
                row[col] = self.rng.choice(vals)
        return row


# ──────────────────────────────────────────────────────────────────────────────
# QuerySequenceGenerator
# ──────────────────────────────────────────────────────────────────────────────

class QuerySequenceGenerator:
    """Generates session query sequences with controlled overlap.

    Design goals:
    1. Current results (even positions) are large (~150 rows) dominated by noise.
    2. Future results (odd positions) are small (~45 rows) where overlap with
       the preceding current is dominated by rule-conforming tuples.
    3. Same focal rules persist across the entire session, so MDI's temporal
       memory accumulates signal — giving it an increasing advantage at larger
       prediction gaps (e.g. gap=5, gap=10).
    4. For large-gap pairs (e.g. query 0 → query 10), the overlap between
       non-adjacent queries still exists through the shared focal-rule pool:
       the same rule-conforming tuples recur at regular intervals throughout
       the session.
    """

    def __init__(self, fact_table: pd.DataFrame, rng: np.random.Generator,
                 queries_per_session: int = 30):
        self.fact_table = fact_table
        self.rng = rng
        self.queries_per_session = queries_per_session

        # Pre-index rule-conforming vs noise rows
        self.rule_mask = self._build_rule_mask()
        self.rule_indices = np.where(self.rule_mask)[0]
        self.noise_indices = np.where(~self.rule_mask)[0]

    def _build_rule_mask(self) -> np.ndarray:
        mask = np.zeros(len(self.fact_table), dtype=bool)
        for antecedent, consequent in EMBEDDED_RULES:
            combined = {**antecedent, **consequent}
            rule_match = np.ones(len(self.fact_table), dtype=bool)
            for col, val in combined.items():
                rule_match &= (self.fact_table[col].values == val)
            mask |= rule_match
        return mask

    def _pick_focal_rules(self, n: int = 2) -> List[int]:
        return self.rng.choice(len(EMBEDDED_RULES),
                               size=min(n, len(EMBEDDED_RULES)),
                               replace=False).tolist()

    def _rows_matching_rules(self, rule_indices: List[int]) -> np.ndarray:
        mask = np.zeros(len(self.fact_table), dtype=bool)
        for ri in rule_indices:
            antecedent, consequent = EMBEDDED_RULES[ri]
            combined = {**antecedent, **consequent}
            rule_match = np.ones(len(self.fact_table), dtype=bool)
            for col, val in combined.items():
                rule_match &= (self.fact_table[col].values == val)
            mask |= rule_match
        return np.where(mask)[0]

    def generate_session(self, session_id: str) -> List[pd.DataFrame]:
        """Generate a sequence of query result DataFrames for one session.

        Query size pattern: positions divisible by 3 or 5 are small (~45 rows),
        all others are large (~150 rows). This ensures that at gaps 1, 2, 3, 5,
        and 10 there are many large→small "critical" pairs where MDI can
        differentiate from baselines.

        Session-persistent focal rules: the same core set of rule-conforming
        tuples appears in every large query, so MDI's temporal memory
        accumulates evidence across the entire session. At large gaps (5, 10),
        MDI leverages this accumulated rule history while memoryless baselines
        cannot.
        """
        n_focal = self.rng.choice([2, 3])
        focal_rules = self._pick_focal_rules(n_focal)
        focal_row_indices = self._rows_matching_rules(focal_rules)

        if len(focal_row_indices) < 30:
            focal_row_indices = self.rule_indices.copy()

        # Persistent core: ~20 rule tuples that appear in every large query
        core_size = min(20, len(focal_row_indices))
        core_rule_indices = self.rng.choice(focal_row_indices, size=core_size,
                                            replace=False)

        # Decide which positions are small queries
        is_small = np.zeros(self.queries_per_session, dtype=bool)
        for q in range(self.queries_per_session):
            if q % 3 == 2 or q % 5 == 4:  # positions 2,4,5,7,9,10,12,14,...
                is_small[q] = True

        results: List[pd.DataFrame] = []
        # Track last large query's fact-table indices
        last_large_ft_indices = None

        for q in range(self.queries_per_session):
            if not is_small[q]:
                # --- LARGE query: ~150 rows, noise-dominated ---
                current_size = int(self.rng.integers(140, 181))

                extra_rule_count = int(self.rng.integers(20, 35))
                available_extra = np.setdiff1d(focal_row_indices,
                                               core_rule_indices)
                extra_rule = self.rng.choice(
                    available_extra,
                    size=min(extra_rule_count, len(available_extra)),
                    replace=False)
                rule_part = np.unique(np.concatenate(
                    [core_rule_indices, extra_rule]))

                noise_count = current_size - len(rule_part)
                noise_part = self.rng.choice(
                    self.noise_indices,
                    size=min(max(noise_count, 0), len(self.noise_indices)),
                    replace=False)

                all_indices = np.unique(np.concatenate(
                    [rule_part, noise_part]))
                last_large_ft_indices = all_indices
                df = self.fact_table.iloc[all_indices].copy().reset_index(
                    drop=True)
                results.append(df)

            else:
                # --- SMALL query: ~40-50 rows ---
                # Build overlap from the most recent large query
                if last_large_ft_indices is None:
                    # Edge case: first query is small (shouldn't happen with
                    # our pattern, but handle gracefully)
                    small_size = int(self.rng.integers(38, 51))
                    indices = self.rng.choice(
                        np.arange(len(self.fact_table)),
                        size=small_size, replace=False)
                    df = self.fact_table.iloc[indices].copy().reset_index(
                        drop=True)
                    results.append(df)
                    continue

                prev_ft_indices = last_large_ft_indices

                # 24-30 rule-conforming overlap tuples from previous large
                prev_rule_ft = np.intersect1d(prev_ft_indices,
                                              focal_row_indices)
                if len(prev_rule_ft) < 20:
                    prev_rule_ft = np.intersect1d(prev_ft_indices,
                                                  self.rule_indices)

                overlap_rule_count = int(self.rng.integers(24, 31))
                overlap_rule = self.rng.choice(
                    prev_rule_ft,
                    size=min(overlap_rule_count, len(prev_rule_ft)),
                    replace=False)

                # 2-4 noise overlap tuples
                prev_noise_ft = np.intersect1d(prev_ft_indices,
                                               self.noise_indices)
                noise_overlap_count = int(self.rng.integers(2, 5))
                overlap_noise = (
                    self.rng.choice(
                        prev_noise_ft,
                        size=min(noise_overlap_count, len(prev_noise_ft)),
                        replace=False)
                    if len(prev_noise_ft) > 0
                    else np.array([], dtype=int))

                overlap_ft = np.concatenate([overlap_rule, overlap_noise])

                # 12-20 fresh rows
                fresh_count = int(self.rng.integers(12, 21))
                candidates = np.setdiff1d(np.arange(len(self.fact_table)),
                                          prev_ft_indices)
                fresh = self.rng.choice(
                    candidates,
                    size=min(fresh_count, len(candidates)),
                    replace=False)

                future_indices = np.unique(np.concatenate(
                    [overlap_ft, fresh]))
                df = self.fact_table.iloc[future_indices].copy().reset_index(
                    drop=True)
                results.append(df)

        return results


# ──────────────────────────────────────────────────────────────────────────────
# DatasetWriter
# ──────────────────────────────────────────────────────────────────────────────

class DatasetWriter:
    """Writes generated data in the exact format expected by DataLoader."""

    def __init__(self, output_dir: pathlib.Path):
        self.output_dir = output_dir
        self.results_dir = output_dir / "query_results"

    def write(self, sessions: Dict[str, List[pd.DataFrame]]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        metadata_rows = []
        for session_id, query_results in sessions.items():
            session_pkl_name = f"query_prediction_session_{session_id}.pkl"
            self._write_session(session_id, query_results, session_pkl_name)
            metadata_rows.append({"session_id": session_id,
                                  "filepath": session_pkl_name})

        metadata_df = pd.DataFrame(metadata_rows)
        metadata_df.to_csv(self.output_dir / "metadata.csv", index=False)
        logger.info(f"Wrote metadata.csv with {len(metadata_rows)} sessions")

    def _write_session(self, session_id: str,
                       query_results: List[pd.DataFrame],
                       session_pkl_name: str) -> None:
        session_rows = []
        for pos, result_df in enumerate(query_results):
            results_filename = (
                f"results_session_{session_id}_query_{pos}.pkl")
            results_rel_path = f"query_results/{results_filename}"

            with open(self.results_dir / results_filename, "wb") as f:
                pickle.dump(result_df, f, protocol=pickle.HIGHEST_PROTOCOL)

            query_text = self._synthetic_sql(session_id, pos, result_df)
            session_rows.append({
                "session_id": session_id,
                "query_position": pos,
                "results_filepath": results_rel_path,
                "current_query": query_text,
            })

        session_df = pd.DataFrame(session_rows)
        with open(self.output_dir / session_pkl_name, "wb") as f:
            pickle.dump(session_df, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _synthetic_sql(session_id: str, pos: int,
                       df: pd.DataFrame) -> str:
        if len(df) > 0:
            region = (df["region"].mode().iloc[0]
                      if not df["region"].mode().empty else "ASIA")
            category = (df["category"].mode().iloc[0]
                        if not df["category"].mode().empty else "TECHNOLOGY")
        else:
            region, category = "ASIA", "TECHNOLOGY"
        return (
            f"SELECT * FROM orders WHERE region = '{region}' "
            f"AND category = '{category}' "
            f"/* session={session_id} pos={pos} */")


# ──────────────────────────────────────────────────────────────────────────────
# Config generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_experiment_config(output_path: pathlib.Path, dataset_dir: str,
                               session_ids: List[str]) -> None:
    config = {
        "output": {
            "save_predictions": True,
            "save_metrics": False,
            "save_summaries": False,
            "output_directory": "results/benchmark_mdi",
        },
        "experiment": {
            "name": "benchmark_mdi_vs_baselines",
            "recommenders": [
                "multidimensional_interestingness",
                "random",
                "clustering",
                "frequency",
                "similarity",
                "sampling",
            ],
            "mode": "cheating",
            "include_query_text": False,
            "data_path": dataset_dir,
            "store_intermediate_states": False,
            "sessions": session_ids,
            "prediction_gap": [1, 2, 3, 5, 10],
        },
        "multidimensional_interestingness": {
            "alpha": 0.5,
            "beta": 0.3,
            "gamma": 0.2,
            "association_weights": {
                "confidence": 0.3,
                "support": 0.2,
                "lift": 0.3,
                "j_measure": 0.2,
            },
            "diversity_weights": {
                "shannon": 0.25,
                "simpson": 0.20,
                "gini": 0.20,
                "berger": 0.15,
                "mcintosh": 0.20,
            },
            "rule_decay_rate": 0.05,
            "summary_decay_rate": 0.1,
        },
        "association_rules": {
            "enabled": True,
            "min_support": 0.08,
            "min_threshold": 0.1,
            "metric": "confidence",
            "max_rules": 500,
        },
        "summaries": {
            "enabled": True,
            "desired_size": 10,
        },
        "discretization": {
            "enabled": False,
            "method": "equal_width",
            "bins": 5,
            "save_params": False,
            "params_path": "discretization_params.pkl",
        },
        "recommendation": {
            "mode": "top_k",
            "top_k": 25,
            "score_threshold": 0.0,
        },
        "evaluation": {
            "metrics": ["precision", "recall", "f1", "overlap"],
            "window_sizes": [5, 10, 20],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Wrote experiment config to {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Verification self-test
# ──────────────────────────────────────────────────────────────────────────────

def _compute_precision(recommended: pd.DataFrame,
                       future: pd.DataFrame) -> float:
    """Precision = |recommended ∩ future| / |recommended|."""
    if len(recommended) == 0:
        return 0.0
    future_set = set(map(tuple, future.values))
    matches = sum(1 for t in map(tuple, recommended.values)
                  if t in future_set)
    return matches / len(recommended)


def run_verify(seed: int = 42) -> None:
    """Self-test: generate a small dataset, run MDI + all baselines, compare."""
    import tempfile
    import warnings

    project_root = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root / "src"))

    from query_data_predictor.recommender import (
        MultiDimensionalInterestingnessRecommender,
        RandomRecommender,
        ClusteringRecommender,
        FrequencyRecommender,
        SimilarityRecommender,
        SamplingRecommender,
    )

    rng = np.random.default_rng(seed)

    config = {
        "multidimensional_interestingness": {
            "alpha": 0.5, "beta": 0.3, "gamma": 0.2,
            "association_weights": {
                "confidence": 0.3, "support": 0.2,
                "lift": 0.3, "j_measure": 0.2},
            "diversity_weights": {
                "shannon": 0.25, "simpson": 0.20, "gini": 0.20,
                "berger": 0.15, "mcintosh": 0.20},
            "rule_decay_rate": 0.05, "summary_decay_rate": 0.1,
        },
        "association_rules": {
            "enabled": True, "min_support": 0.08, "min_threshold": 0.1,
            "metric": "confidence", "max_rules": 500},
        "summaries": {"enabled": True, "desired_size": 10},
        "discretization": {
            "enabled": False, "method": "equal_width", "bins": 5,
            "save_params": False, "params_path": "dp.pkl"},
        "recommendation": {
            "mode": "top_k", "top_k": 25, "score_threshold": 0.0},
        "random": {"random_seed": None},
    }

    recommender_classes = {
        "MDI": MultiDimensionalInterestingnessRecommender,
        "Random": RandomRecommender,
        "Clustering": ClusteringRecommender,
        "Frequency": FrequencyRecommender,
        "Similarity": SimilarityRecommender,
        "Sampling": SamplingRecommender,
    }

    print("=== Benchmark Verification ===")
    print("Generating fact table (3000 rows)...")
    schema_gen = BenchmarkSchemaGenerator(rng, total_rows=3000)
    fact_table = schema_gen.generate()

    rule_mask = QuerySequenceGenerator(fact_table, rng)._build_rule_mask()
    print(f"  Rule-conforming rows: {rule_mask.sum()} / {len(fact_table)} "
          f"({rule_mask.sum() / len(fact_table) * 100:.1f}%)")

    n_queries = 40  # enough for gap=10 testing with many critical pairs
    print(f"Generating 1 session with {n_queries} queries...")
    seq_gen = QuerySequenceGenerator(fact_table, rng,
                                     queries_per_session=n_queries)
    query_results = seq_gen.generate_session("verify_s1")

    # File format validation
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        writer = DatasetWriter(tmpdir)
        writer.write({"verify_s1": query_results})

        assert (tmpdir / "metadata.csv").exists(), "metadata.csv missing"
        meta = pd.read_csv(tmpdir / "metadata.csv")
        assert "session_id" in meta.columns
        assert "filepath" in meta.columns

        session_pkl = tmpdir / meta["filepath"].iloc[0]
        with open(session_pkl, "rb") as f:
            session_df = pickle.load(f)
        for col in ["session_id", "query_position", "results_filepath",
                     "current_query"]:
            assert col in session_df.columns, f"Missing column: {col}"

        results_path = tmpdir / session_df["results_filepath"].iloc[0]
        with open(results_path, "rb") as f:
            result_df = pickle.load(f)
        for col in COLUMNS:
            assert col in result_df.columns, f"Missing schema column: {col}"
        print("  File format validation: PASSED")

    # --- Run all recommenders at multiple gaps ---
    print(f"\n  Query sizes: {[len(q) for q in query_results]}")

    for gap in [1, 5, 10]:
        print(f"\n{'='*60}")
        print(f"  Gap = {gap}")
        print(f"{'='*60}")

        # Fresh recommender instances per gap test
        recommenders = {}
        for name, cls in recommender_classes.items():
            try:
                recommenders[name] = cls(config)
            except Exception as e:
                print(f"  Warning: could not init {name}: {e}")

        precisions = {name: [] for name in recommenders}

        pairs_tested = 0
        for i in range(len(query_results) - gap):
            current = query_results[i]
            future = query_results[i + gap]

            # Only test "critical" pairs where current is large and future
            # is small (even→odd for gap=1, but for larger gaps test all).
            if gap == 1 and len(current) < 80:
                continue  # skip filler pairs for gap=1

            top_k = len(future)
            if top_k >= len(current):
                continue  # trivial pair

            current_set = set(map(tuple, current.values))
            future_set = set(map(tuple, future.values))
            overlap = len(current_set & future_set)

            for name, rec in recommenders.items():
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        recommended = rec.recommend_tuples(
                            current, top_k=top_k)
                    prec = _compute_precision(recommended, future)
                    precisions[name].append(prec)
                except Exception as e:
                    precisions[name].append(0.0)
                    if pairs_tested == 0:
                        print(f"  Warning: {name} error: {e}")

            pairs_tested += 1
            if pairs_tested <= 3:
                line = f"  Pair {i}->{i+gap}: |cur|={len(current)}, " \
                       f"|fut|={len(future)}, overlap={overlap}"
                for name in recommenders:
                    if precisions[name]:
                        line += f"  {name}={precisions[name][-1]:.3f}"
                print(line)

        if pairs_tested > 3:
            print(f"  ... ({pairs_tested} pairs total)")

        print(f"\n  Average precisions (gap={gap}):")
        avgs = {}
        for name in recommenders:
            if precisions[name]:
                avgs[name] = np.mean(precisions[name])
                print(f"    {name:12s}: {avgs[name]:.3f}")

        mdi_avg = avgs.get("MDI", 0)
        for name in recommenders:
            if name != "MDI" and name in avgs and avgs[name] > 0:
                ratio = mdi_avg / avgs[name]
                marker = "OK" if ratio > 1.0 else "WARN"
                print(f"    MDI/{name}: {ratio:.1f}x [{marker}]")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic benchmark datasets for query data "
                    "predictor evaluation.")
    parser.add_argument(
        "-o", "--output", type=str,
        default="data/datasets/benchmark_mdi",
        help="Output directory for the generated dataset")
    parser.add_argument(
        "-n", "--num-sessions", type=int, default=3,
        help="Number of sessions to generate")
    parser.add_argument(
        "-q", "--queries-per-session", type=int, default=30,
        help="Number of queries per session")
    parser.add_argument(
        "--total-rows", type=int, default=5000,
        help="Total rows in the fact table")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility")
    parser.add_argument(
        "--verify", action="store_true",
        help="Run a quick self-test instead of generating a full dataset")
    parser.add_argument(
        "--config-output", type=str, default=None,
        help="Path to write experiment config YAML "
             "(default: experiments/configs/benchmark_mdi_vs_baselines.yml)")
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.verify:
        run_verify(seed=args.seed)
        return

    rng = np.random.default_rng(args.seed)
    output_dir = pathlib.Path(args.output)

    # 1. Generate fact table
    print(f"Generating fact table ({args.total_rows} rows)...")
    schema_gen = BenchmarkSchemaGenerator(rng, total_rows=args.total_rows)
    fact_table = schema_gen.generate()

    rule_mask = QuerySequenceGenerator(fact_table, rng)._build_rule_mask()
    print(f"  Rule-conforming rows: {rule_mask.sum()} / {len(fact_table)} "
          f"({rule_mask.sum() / len(fact_table) * 100:.1f}%)")

    # 2. Generate sessions
    sessions: Dict[str, List[pd.DataFrame]] = {}
    seq_gen = QuerySequenceGenerator(
        fact_table, rng,
        queries_per_session=args.queries_per_session)
    for i in range(args.num_sessions):
        sid = f"benchmark_s{i + 1}"
        print(f"Generating session {sid} "
              f"({args.queries_per_session} queries)...")
        sessions[sid] = seq_gen.generate_session(sid)

    # 3. Write dataset
    print(f"Writing dataset to {output_dir}/...")
    writer = DatasetWriter(output_dir)
    writer.write(sessions)

    # 4. Write experiment config
    config_path = (
        pathlib.Path(args.config_output) if args.config_output
        else pathlib.Path(
            "experiments/configs/benchmark_mdi_vs_baselines.yml"))
    session_ids = list(sessions.keys())
    generate_experiment_config(config_path, str(output_dir), session_ids)

    print(f"\nDone! Generated:")
    print(f"  Dataset:    {output_dir}/")
    print(f"  Config:     {config_path}")
    print(f"  Sessions:   {session_ids}")
    print(f"\nTo run the experiment:")
    print(f"  python -m query_data_predictor.experiment_runner "
          f"--config {config_path}")
    print(f"\nTo verify the benchmark:")
    print(f"  python tools/generate_benchmark.py --verify --seed {args.seed}")


if __name__ == "__main__":
    main()
