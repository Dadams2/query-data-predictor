"""
Extract paper table values from the analysis CSVs used by the vision paper.

By default this script reads `paper_artifact_manifest.yaml`, resolves the
preferred paper result snapshots, and writes reviewer-facing outputs under
`artifacts/paper/tables/`.

Usage:
    UV_CACHE_DIR=.uv-cache MPLCONFIGDIR=.matplotlib XDG_CACHE_HOME=.cache \
      uv run python extract_table_data.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from query_data_predictor.paper_artifact import (
    DEFAULT_MANIFEST_PATH,
    aggregate_gap_metrics,
    load_manifest,
    load_metrics_dataframe,
    resolve_experiment_paths,
    to_repo_relative,
)


METRIC_COLUMNS = ["precision", "recall", "f1_score", "overlap"]


def _format_metric(value: float) -> str:
    return f"{value:.3f}"


def _format_latex_decimal(value: float) -> str:
    return f"{value:.3f}".replace("0.", ".")


def table_payload(
    manifest: dict[str, Any],
    table_key: str,
    prefer_frozen: bool = True,
) -> dict[str, Any]:
    """Build the full data payload for one paper table."""
    table = manifest["tables"][table_key]
    experiment_paths = resolve_experiment_paths(
        manifest, table["experiment"], prefer_frozen=prefer_frozen
    )
    df = load_metrics_dataframe(experiment_paths)
    agg = aggregate_gap_metrics(df, table["gap"])

    rows = []
    for recommender in table["order"]:
        row = agg.loc[recommender]
        rows.append(
            {
                "recommender": recommender,
                "display_name": table["display_names"][recommender],
                "precision": round(float(row["precision"]), 3),
                "recall": round(float(row["recall"]), 3),
                "f1_score": round(float(row["f1_score"]), 3),
                "overlap": round(float(row["overlap"]), 3),
            }
        )

    return {
        "table_key": table_key,
        "label": table["label"],
        "gap": table["gap"],
        "experiment_label": experiment_paths.label,
        "analysis_csv": to_repo_relative(experiment_paths.analysis_csv),
        "results_dir": to_repo_relative(experiment_paths.results_dir),
        "config": to_repo_relative(experiment_paths.config_path),
        "rows": rows,
    }


def write_outputs(output_dir: Path, payloads: list[dict[str, Any]]):
    """Write JSON, Markdown, and LaTeX table summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "paper_tables.json"
    markdown_path = output_dir / "paper_tables.md"
    latex_path = output_dir / "paper_table_rows.tex"

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payloads, handle, indent=2)
        handle.write("\n")

    markdown_lines = ["# Paper Tables", ""]
    latex_lines = []

    for payload in payloads:
        markdown_lines.append(f"## {payload['label']}")
        markdown_lines.append("")
        markdown_lines.append(f"- Experiment: `{payload['experiment_label']}`")
        markdown_lines.append(f"- Config: `{payload['config']}`")
        markdown_lines.append(f"- Results: `{payload['results_dir']}`")
        markdown_lines.append(f"- Analysis CSV: `{payload['analysis_csv']}`")
        markdown_lines.append("")
        markdown_lines.append("| Recommender | Precision | Recall | F1 | Overlap |")
        markdown_lines.append("| --- | ---: | ---: | ---: | ---: |")

        latex_lines.append(f"% {payload['label']}")
        for row in payload["rows"]:
            markdown_lines.append(
                f"| {row['display_name']} | {_format_metric(row['precision'])} | "
                f"{_format_metric(row['recall'])} | {_format_metric(row['f1_score'])} | "
                f"{_format_metric(row['overlap'])} |"
            )
            latex_lines.append(
                f"{row['display_name']} & {_format_latex_decimal(row['precision'])} & "
                f"{_format_latex_decimal(row['recall'])} & {_format_latex_decimal(row['f1_score'])} & "
                f"{_format_latex_decimal(row['overlap'])} \\\\"
            )
        markdown_lines.append("")
        latex_lines.append("")

    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    latex_path.write_text("\n".join(latex_lines), encoding="utf-8")

    return json_path, markdown_path, latex_path


def check_expected(manifest: dict[str, Any], payloads: list[dict[str, Any]]) -> list[str]:
    """Compare computed paper metrics against the manifest's expected values."""
    mismatches: list[str] = []

    for payload in payloads:
        expected_rows = manifest["tables"][payload["table_key"]]["expected"]
        for row in payload["rows"]:
            expected = expected_rows[row["recommender"]]
            for metric in METRIC_COLUMNS:
                actual_value = round(float(row[metric]), 3)
                expected_value = round(float(expected[metric]), 3)
                if actual_value != expected_value:
                    mismatches.append(
                        f"{payload['table_key']}::{row['recommender']}::{metric} "
                        f"expected {expected_value:.3f} got {actual_value:.3f}"
                    )

    return mismatches


def main():
    parser = argparse.ArgumentParser(description="Extract paper table values")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/paper/tables"))
    parser.add_argument("--check", action="store_true", help="Verify metrics against manifest expectations")
    parser.add_argument("--use-latest-results", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    table_keys = manifest["paper"]["active_outputs"]["tables"]
    prefer_frozen = not args.use_latest_results
    payloads = [
        table_payload(manifest, table_key, prefer_frozen=prefer_frozen)
        for table_key in table_keys
    ]

    json_path, markdown_path, latex_path = write_outputs(args.output_dir, payloads)

    print("Wrote paper table outputs:")
    print(f"  - {json_path}")
    print(f"  - {markdown_path}")
    print(f"  - {latex_path}")

    for payload in payloads:
        print(f"\n{payload['label']}")
        for row in payload["rows"]:
            print(
                f"  {row['display_name']}: "
                f"P={row['precision']:.3f}, R={row['recall']:.3f}, "
                f"F1={row['f1_score']:.3f}, Ovlp={row['overlap']:.3f}"
            )

    if args.check:
        mismatches = check_expected(manifest, payloads)
        if mismatches:
            print("\nVerification failed:")
            for mismatch in mismatches:
                print(f"  - {mismatch}")
            raise SystemExit(1)
        print("\nVerification passed: computed values match the manifest expectations.")


if __name__ == "__main__":
    main()
