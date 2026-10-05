"""Audit the SkyServer session snapshots quoted in the paper (Section 4, Benchmarks).

For each consecutive pair of query results (gap 1), compares the sets of exact
full rows (including column names). Reports, per session and pooled, how often a
result shares no rows with the next one and the median result size. Read-only.

Run: python tools/audit_sdss.py
"""
import json
from pathlib import Path
from statistics import mean, median

SNAPSHOTS = Path(__file__).resolve().parents[1] / 'data' / 'sdss_snapshots'


def rows(values):
    if not isinstance(values, list):
        raise ValueError('Missing result snapshot')
    return frozenset(tuple(sorted(row.items())) for row in values)


def audit():
    transitions = {}
    files = sorted(SNAPSHOTS.glob('*__gap-1.json'))
    assert files, f'No snapshots in {SNAPSHOTS}'
    for path in files:
        for record in json.loads(path.read_text()):
            assert record['gap'] == 1
            key = (record['session_id'], record['current_query_id'], record['future_query_id'])
            pair = (rows(record['current_results']), rows(record['future_results']))
            # Snapshots were stored once per recommender; they must agree.
            if key in transitions:
                assert transitions[key] == pair, (path, key)
            transitions[key] = pair
    sessions = []
    for sid in sorted({key[0] for key in transitions}):
        pairs = [pair for key, pair in transitions.items() if key[0] == sid]
        sessions.append(dict(
            session=sid, transitions=len(pairs),
            zero_overlap=mean(not (a & b) for a, b in pairs),
            median_result_rows=median(len(a) for a, b in pairs)))
    total = sum(s['transitions'] for s in sessions)
    pooled = sum(s['zero_overlap'] * s['transitions'] for s in sessions) / total
    return dict(
        sessions=len(sessions), transitions=total,
        pooled_zero_overlap=round(pooled, 3),
        zero_overlap_range=[round(min(s['zero_overlap'] for s in sessions), 3),
                            round(max(s['zero_overlap'] for s in sessions), 3)],
        sessions_with_single_row_median=sum(s['median_result_rows'] == 1 for s in sessions),
        per_session=sessions)


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
