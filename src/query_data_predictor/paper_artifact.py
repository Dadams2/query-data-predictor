"""Helpers for paper artifact setup, result discovery, and verification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from query_data_predictor.analysis import ResultsAnalyzer


DEFAULT_MANIFEST_PATH = Path("paper_artifact_manifest.yaml")


@dataclass
class ExperimentPaths:
    """Resolved file system paths for one experiment used by the paper."""

    name: str
    config_path: Path
    results_dir: Path
    analysis_csv: Path
    label: str


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_manifest(manifest_path: Path | None = None) -> dict[str, Any]:
    """Load the paper artifact manifest."""
    target = Path(manifest_path or DEFAULT_MANIFEST_PATH)
    return load_yaml(target)


def discover_latest_results_dir(config_path: Path) -> Path:
    """
    Find the most recent timestamped results directory for a config.

    This uses the config's output_directory plus experiment.name prefix so the
    artifact workflow can survive new reruns without editing Python constants.
    """
    config = load_yaml(config_path)
    output_dir = Path(config["output"]["output_directory"])
    experiment_name = config["experiment"]["name"]

    if not output_dir.exists():
        raise FileNotFoundError(f"Configured output directory does not exist: {output_dir}")

    candidates = sorted(
        [
            path
            for path in output_dir.iterdir()
            if path.is_dir() and path.name.startswith(f"{experiment_name}_")
        ],
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            f"No timestamped results found in {output_dir} for experiment {experiment_name}"
        )

    return candidates[0]


def resolve_experiment_paths(
    manifest: dict[str, Any],
    experiment_key: str,
    prefer_frozen: bool = True,
) -> ExperimentPaths:
    """Resolve config, result directory, and analysis CSV for an experiment entry."""
    experiment = manifest["experiments"][experiment_key]
    config_path = Path(experiment["config"])

    preferred_results_dir = Path(experiment.get("preferred_results_dir", ""))
    latest_results_dir: Path | None = None
    if prefer_frozen:
        if preferred_results_dir and preferred_results_dir.exists():
            results_dir = preferred_results_dir
        else:
            results_dir = discover_latest_results_dir(config_path)
    else:
        latest_results_dir = discover_latest_results_dir(config_path)
        results_dir = latest_results_dir
        if (not results_dir.exists()) and preferred_results_dir.exists():
            results_dir = preferred_results_dir

    preferred_csv = Path(experiment.get("preferred_analysis_csv", ""))
    latest_csv = results_dir / "analysis" / "summary" / "raw" / "all_sessions_metrics.csv"

    if prefer_frozen:
        if preferred_csv and preferred_csv.exists():
            analysis_csv = preferred_csv
        else:
            analysis_csv = latest_csv
    else:
        analysis_csv = latest_csv
        if not analysis_csv.exists() and preferred_csv.exists():
            analysis_csv = preferred_csv

    return ExperimentPaths(
        name=experiment_key,
        config_path=config_path,
        results_dir=results_dir,
        analysis_csv=analysis_csv,
        label=experiment.get("label", experiment_key),
    )


def ensure_analysis_csv(paths: ExperimentPaths) -> Path:
    """Create the simple-analysis CSV if it is missing."""
    if paths.analysis_csv.exists():
        return paths.analysis_csv

    analyzer = ResultsAnalyzer(
        results_dir=paths.results_dir,
        config=load_yaml(paths.config_path),
    )
    analyzer.analyze_simple()

    if not paths.analysis_csv.exists():
        raise FileNotFoundError(f"Expected analysis CSV was not created: {paths.analysis_csv}")

    return paths.analysis_csv


def load_metrics_dataframe(paths: ExperimentPaths) -> pd.DataFrame:
    """Resolve and load the paper-facing metrics CSV for one experiment."""
    csv_path = ensure_analysis_csv(paths)
    return pd.read_csv(csv_path)


def aggregate_gap_metrics(
    df: pd.DataFrame,
    gap: int,
) -> pd.DataFrame:
    """Aggregate paper table metrics for one prediction gap."""
    filtered = df[df["gap"] == gap]
    if filtered.empty:
        raise ValueError(f"No rows found for gap={gap}")

    return (
        filtered.groupby("recommender")[["precision", "recall", "f1_score", "overlap"]]
        .mean()
        .round(3)
    )


def to_repo_relative(path: Path) -> str:
    """Render a path relative to the repository when possible."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)
