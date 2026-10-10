"""Characterise the SkyServer (SDSS) session log from query text and result sizes.

Read-only: uses data/skyserver_sessions.csv.gz, which holds, for each of 462
SkyServer SQL sessions, every query's text, type and result size (result rows
are not needed), and writes a JSON summary. Run from the repository root:

    bash scripts/download_skyserver_extract.sh
    .venv/bin/python tools/sdss_characterise.py --output artifacts/sdss-characterisation

Definitions (deliberately simple, regex-based):
- id lookup: an equality predicate on an object/spectrum/field identifier, or
  a nearby/rectangle object function (fGetNearbyObj, fGetObjFromRect, ...).
- single-row: the query returned at most one row.
- template: the query with numeric and string literals replaced by '?'.
  A query is templated if its template occurs in at least --template-sessions
  distinct sessions.
- refinement step: consecutive SELECTs over the same tables whose predicate
  columns are a superset of the previous query's, whose text differs, whose
  result has more than one row and is no larger than the previous one.
- strict refinement: a refinement step that adds at least one predicate
  column (a constant-only change, such as moving a coordinate window, is not
  strict).
- trajectory-like session: contains a run of at least --chain refinement steps
  (reported for both definitions).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from statistics import median

import pandas as pd

ID_COLUMN = re.compile(r'\b(?:[a-z]\w*\.)?(?:obj|specobj|bestobj|field|plate|photoobj|run)id\s*=\s*', re.I)
OBJECT_FUNCTION = re.compile(r'\bf(?:GetNearbyObj\w*|GetNearestObj\w*|GetObjFromRect\w*|GetObjectsEq\w*)\s*\(', re.I)
LITERAL = re.compile(r"'(?:[^']|'')*'|\b0x[0-9a-f]+\b|-?\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b", re.I)
CLAUSE_END = r'(?=\bgroup\s+by\b|\border\s+by\b|\bhaving\b|\bunion\b|$)'
FROM = re.compile(r'\bfrom\b(.*?)(?=\bwhere\b|\bgroup\s+by\b|\border\s+by\b|\bhaving\b|\bunion\b|$)', re.I | re.S)
WHERE = re.compile(r'\bwhere\b(.*?)' + CLAUSE_END, re.I | re.S)
PREDICATE = re.compile(r'([a-z_][\w.]*)\s*(?:<=|>=|<>|!=|=|<|>|\bbetween\b|\blike\b|\bin\b)', re.I)


def template(query: str) -> str:
    return re.sub(r'\s+', ' ', LITERAL.sub('?', query.lower())).strip()


def tables(query: str) -> frozenset:
    match = FROM.search(query)
    if not match:
        return frozenset()
    parts = re.split(r',|\bjoin\b', match.group(1), flags=re.I)
    names = set()
    for part in parts:
        tokens = part.strip().split()
        if tokens and tokens[0].lower() not in ('on', '('):
            names.add(tokens[0].lower().split('.')[-1])
    return frozenset(names)


def predicate_columns(query: str) -> frozenset:
    match = WHERE.search(query)
    if not match:
        return frozenset()
    return frozenset(col.lower().split('.')[-1] for col in PREDICATE.findall(match.group(1))
                     if col.lower() not in ('and', 'or', 'not'))


def is_refinement(prev: dict, curr: dict) -> bool:
    return bool(prev['select'] and curr['select'] and prev['tables'] and prev['tables'] == curr['tables']
            and prev['predicates'] and curr['predicates'] >= prev['predicates']
            and curr['text'] != prev['text']
            and 1 < curr['rows'] <= prev['rows'])


def is_strict_refinement(prev: dict, curr: dict) -> bool:
    return is_refinement(prev, curr) and curr['predicates'] > prev['predicates']


def longest_run(flags: list[bool]) -> int:
    best = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default='data/skyserver_sessions.csv.gz')
    parser.add_argument('--output', type=Path, default=Path('artifacts/sdss-characterisation'))
    parser.add_argument('--template-sessions', type=int, default=5)
    parser.add_argument('--chain', type=int, default=3)
    args = parser.parse_args()

    log = pd.read_csv(args.data, dtype={'current_query': str, 'query_type': str}, keep_default_na=False)
    log['result_row_count'] = pd.to_numeric(log.result_row_count, errors='coerce')
    sessions = {}
    for _, frame in log.groupby('session_id', sort=False):
        frame = frame.sort_values('query_position')
        queries = []
        for _, row in frame.iterrows():
            text = str(row.current_query)
            queries.append(dict(
                text=text.strip(), template=template(text),
                select=str(row.query_type).upper() == 'SELECT',
                tables=tables(text), predicates=predicate_columns(text),
                rows=int(row.result_row_count) if pd.notna(row.result_row_count) else 0,
                id_lookup=bool(ID_COLUMN.search(text) or OBJECT_FUNCTION.search(text))))
        sessions[str(frame.session_id.iloc[0])] = queries

    template_sessions = defaultdict(set)
    for sid, queries in sessions.items():
        for q in queries:
            template_sessions[q['template']].add(sid)

    all_queries = [q for queries in sessions.values() for q in queries]
    n = len(all_queries)
    per_session = []
    for sid, queries in sessions.items():
        steps = [is_refinement(a, b) for a, b in zip(queries, queries[1:])]
        strict = [is_strict_refinement(a, b) for a, b in zip(queries, queries[1:])]
        per_session.append(dict(
            session=sid, length=len(queries),
            refinement_steps=sum(steps), longest_chain=longest_run(steps),
            strict_steps=sum(strict), longest_strict_chain=longest_run(strict),
            templates=len({q['template'] for q in queries}),
            median_rows=median(q['rows'] for q in queries) if queries else 0,
            lookup_share=sum(q['id_lookup'] or q['rows'] <= 1 for q in queries) / max(len(queries), 1)))
    sessions_frame = pd.DataFrame(per_session)
    transitions = sum(max(len(q) - 1, 0) for q in sessions.values())
    qualifying = sessions_frame[sessions_frame.longest_chain >= args.chain]
    strict_qualifying = sessions_frame[sessions_frame.longest_strict_chain >= args.chain]
    strict_two = sessions_frame[sessions_frame.longest_strict_chain >= 2]

    summary = dict(
        sessions=len(sessions), queries=n, transitions=transitions,
        session_length=dict(median=float(sessions_frame.length.median()),
                            p25=float(sessions_frame.length.quantile(.25)),
                            p75=float(sessions_frame.length.quantile(.75)),
                            max=int(sessions_frame.length.max())),
        share_select=sum(q['select'] for q in all_queries) / n,
        share_id_lookup=sum(q['id_lookup'] for q in all_queries) / n,
        share_single_row=sum(q['rows'] <= 1 for q in all_queries) / n,
        share_lookup_or_single_row=sum(q['id_lookup'] or q['rows'] <= 1 for q in all_queries) / n,
        distinct_templates=len(template_sessions),
        template_sessions_threshold=args.template_sessions,
        share_templated=sum(len(template_sessions[q['template']]) >= args.template_sessions
                            for q in all_queries) / n,
        share_refinement_transitions=float(sessions_frame.refinement_steps.sum() / max(transitions, 1)),
        chain_threshold=args.chain,
        sessions_with_chain=len(qualifying),
        share_sessions_with_chain=len(qualifying) / len(sessions),
        queries_in_chain_sessions=int(qualifying.length.sum()),
        share_strict_transitions=float(sessions_frame.strict_steps.sum() / max(transitions, 1)),
        sessions_with_strict_chain=len(strict_qualifying),
        share_sessions_with_strict_chain=len(strict_qualifying) / len(sessions),
        sessions_with_strict_chain_2=len(strict_two),
        sessions_with_any_strict_step=int((sessions_frame.strict_steps > 0).sum()),
        median_templates_per_session=float(sessions_frame.templates.median()),
        longest_strict_chain_distribution=Counter(int(v) for v in sessions_frame.longest_strict_chain).most_common(),
        longest_chain_distribution=Counter(int(v) for v in sessions_frame.longest_chain).most_common(),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    sessions_frame.to_csv(args.output / 'sessions.csv', index=False)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
