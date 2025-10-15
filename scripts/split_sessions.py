#!/usr/bin/env python3
"""Simplified splitter: read with sep="$" and split by session_id.

This script reads `data/SQL_workload1.csv` using `sep="$"`, expects a
`session_id` column and a `statement` column containing the SQL text. It
writes one CSV per session to `data/raw_sessions/` with a single column
named `query`.

Usage:
    python scripts/split_sessions.py

You can override input and output paths with --input and --outdir.
"""
import argparse
import os
from pathlib import Path
import pandas as pd
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="Split SQL_workload1.csv into per-session CSV files")
    parser.add_argument('--input', '-i', default='data/SQL_workload1.csv', help='Input CSV file')
    parser.add_argument('--outdir', '-o', default='data/raw_sessions', help='Output directory for per-session CSVs')
    parser.add_argument('--session-cols', nargs='*', help='Optional list of column names to try for session id (in order)')
    parser.add_argument('--query-cols', nargs='*', help='Optional list of column names to try for query text (in order)')
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    out_dir = Path(args.outdir)

    if not input_path.exists():
        print(f"Input file {input_path} does not exist", file=sys.stderr)
        sys.exit(2)

    out_dir.mkdir(parents=True, exist_ok=True)


    # Read using the requested separator and expected columns
    df = pd.read_csv(input_path, sep="$")

    session_col = 'session_id'
    query_col = 'statement'

    if session_col not in df.columns or query_col not in df.columns:
        print(f"Required columns not found. Expected 'session_id' and 'statement' in {input_path}", file=sys.stderr)
        print(f"Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(3)

    # Check if time column exists
    time_col = 'theTime'
    has_time = time_col in df.columns
    
    grouped = df.groupby(session_col)
    written = 0
    for session_id, group in grouped:
        # Include time column if it exists
        cols_to_keep = [query_col]
        if has_time:
            cols_to_keep.insert(0, time_col)
        
        out_df = group[cols_to_keep].copy()
        out_df = out_df.rename(columns={query_col: 'query'})

        safe_session = str(session_id).replace('/', '_')
        out_path = out_dir / f"session_{safe_session}.csv"
        # mode='w' explicitly overwrites existing files (default behavior)
        out_df.to_csv(out_path, index=False, mode='w')
        written += 1

    print(f"Wrote {written} session files to {out_dir}")


if __name__ == '__main__':
    main()
