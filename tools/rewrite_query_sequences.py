#!/usr/bin/env python3
import pickle
from pathlib import Path

import pandas as pd

from query_data_predictor.path_utils import resolve_stored_path


def mode_value(series):
    m = series.mode(dropna=True)
    return "" if len(m) == 0 else str(m.iloc[0])


def main():
    base = resolve_stored_path(Path("data/datasets/benchmark_mdi"))
    meta = pd.read_csv(base / "metadata.csv")

    for _, row in meta.iterrows():
        sid = row["session_id"]
        session_pkl = resolve_stored_path(row["filepath"], anchor_dir=base)
        with open(session_pkl, "rb") as f:
            session_df = pickle.load(f)

        session_df = session_df.sort_values("query_position")

        out_lines = [
            f"Session ID: {sid}",
            f"Source: {session_pkl.name}",
            f"Queries: {len(session_df)}",
            "Note: current_query in this dataset stores only region/category predicates by design.",
            "",
        ]

        for _, qrow in session_df.iterrows():
            pos = int(qrow["query_position"])
            stored_query = str(qrow["current_query"])
            results_rel = str(qrow["results_filepath"])

            results_path = resolve_stored_path(results_rel, anchor_dir=base)
            with open(results_path, "rb") as rf:
                result_df = pickle.load(rf)

            predicates = [
                f"{col} = '{mode_value(result_df[col])}'"
                for col in result_df.columns
            ]
            inferred_query = "SELECT * FROM orders WHERE " + " AND ".join(predicates)

            out_lines.append(f"[{pos}] STORED_QUERY: {stored_query}")
            out_lines.append(f"[{pos}] RESULTS_FILE: {results_rel}")
            out_lines.append(f"[{pos}] INFERRED_FULL_QUERY: {inferred_query}")
            out_lines.append("")

        (base / f"query_sequence_{sid}.txt").write_text(
            "\n".join(out_lines), encoding="utf-8"
        )
        print(f"updated query_sequence_{sid}.txt")


if __name__ == "__main__":
    main()
