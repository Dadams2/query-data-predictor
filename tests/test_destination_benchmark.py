import pandas as pd
import pytest

import numpy as np

from query_data_predictor.destination_benchmark import (
    CONDITIONS, CONTEXTS, KINDS, TRAJECTORY_LENGTH, SWITCH_STEP, PredicateMatch, Recurrence,
    context_ceiling, context_mask, distinguishing_step, earliness, extension_mask, make_model, make_session,
    make_table, precision_at_budget,
)
from query_data_predictor.recommender.multidimensional_interestingness_recommender import (
    MultiDimensionalInterestingnessRecommender,
)


def test_table_is_deterministic_and_seed_dependent():
    a, meta_a = make_table(999)
    b, meta_b = make_table(999)
    pd.testing.assert_frame_equal(a, b)
    assert meta_a == meta_b
    assert not a.equals(make_table(998)[0])


@pytest.mark.parametrize('kind', KINDS)
def test_planted_destinations_are_stronger_in_context(kind):
    table, meta = make_table(999)
    for destination in (0, 1):
        in_context = context_mask(table, CONTEXTS[kind][destination])
        ext = extension_mask(table, kind, destination, meta)
        assert ext[~in_context].sum() == 0
        rate = ext[in_context].mean()
        assert rate > .15
    overlap = extension_mask(table, kind, 0, meta) & extension_mask(table, kind, 1, meta)
    if kind == 'shared':
        # One context: extensions may overlap in the table; the panel uses exclusive rows.
        assert CONTEXTS[kind][0] == CONTEXTS[kind][1]
    else:
        assert not overlap.any()


@pytest.mark.parametrize('kind', KINDS)
def test_panel_targets_are_disjoint_and_context_ceiling_is_known(kind):
    session = make_session(999, kind, 'clear', 'holdout', 0)
    _, meta = make_table(999)
    e0, e1 = (extension_mask(session.panel, kind, d, meta) for d in (0, 1))
    assert not (e0 & e1).any()
    expected = .3 if kind == 'shared' else .6
    assert context_ceiling(session.panel, kind, 0, meta) == pytest.approx(expected)


def test_distinguishing_steps():
    _, meta = make_table(999)
    assert [distinguishing_step('structural', c, meta) for c in CONDITIONS] == [2, 6, 2, 5, None]
    assert [distinguishing_step('shared', c, meta) for c in CONDITIONS] == [3, 7, 3, 6, None]


def test_context_oracle_reaches_the_ceiling_and_no_further():
    _, meta = make_table(999)
    for kind in KINDS:
        session = make_session(999, kind, 'clear', 'holdout', 0)
        ext = set(session.panel.index[extension_mask(session.panel, kind, 0, meta)])
        oracle = make_model('context_oracle', 999, session)
        precisions = []
        for probe in range(TRAJECTORY_LENGTH + 1):
            if probe:
                oracle.observe(session.results[probe - 1], session.queries[probe - 1])
            precisions.append(precision_at_budget(oracle.rank(session.panel, 25), session.panel, ext, 25))
        ceiling = context_ceiling(session.panel, kind, 0, meta)
        # With k = 25 the in-context block is fully or nearly fully selected.
        assert max(precisions) <= max(ceiling, 15 / 25) + 1e-9


@pytest.mark.parametrize('method', ['competing_models', 'value_profile'])
def test_history_baselines_are_deterministic_and_probes_do_not_mutate(method):
    session = make_session(999, 'shared', 'clear', 'holdout', 0)
    a, b = make_model(method, 999), make_model(method, 999)
    for rows, query in zip(session.results[:4], session.queries[:4]):
        a.observe(rows, query)
        b.observe(rows, query)
    first = list(a.rank(session.panel, 10).index)
    assert first == list(a.rank(session.panel, 10).index) == list(b.rank(session.panel, 10).index)
    assert len(a.results) == 4


def test_history_baselines_follow_the_pattern_step_in_a_shared_context():
    _, meta = make_table(999)
    for target in (0, 1):
        session = make_session(999, 'shared', 'clear', 'holdout', target)
        ext = set(session.panel.index[extension_mask(session.panel, 'shared', target, meta)])
        model = make_model('competing_models', 999)
        for rows, query in zip(session.results, session.queries):
            model.observe(rows, query)
        assert precision_at_budget(model.rank(session.panel, 10), session.panel, ext, 10) > .3


def test_novelty_is_monotone_in_frequency():
    model = MultiDimensionalInterestingnessRecommender({'random': {'random_seed': 0}})
    frame = pd.DataFrame({'a': ['new', 'once', 'twice']})
    model._attribute_value_frequencies = {'a': {'new': 0, 'once': 1, 'twice': 2}}
    scores = model._compute_novelty_component(frame)
    assert scores.is_monotonic_decreasing and np.isfinite(scores).all()


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('condition', CONDITIONS)
def test_paired_sessions_share_the_evaluation_result(kind, condition):
    a = make_session(999, kind, condition, 'holdout', 0)
    b = make_session(999, kind, condition, 'holdout', 1)
    pd.testing.assert_frame_equal(a.panel, b.panel)
    assert a.panel.index.is_unique and len(a.queries) == len(a.results) == TRAJECTORY_LENGTH
    _, meta = make_table(999)
    for d in (0, 1):
        assert extension_mask(a.panel, kind, d, meta).sum() == 15
    if condition == 'null':
        for x, y in zip(a.results, b.results):
            pd.testing.assert_frame_equal(x, y)
    else:
        assert a.queries != b.queries


def test_holdout_excludes_evaluation_entities_from_history():
    session = make_session(999, 'structural', 'clear', 'holdout', 0)
    seen = set().union(*(set(r.index) for r in session.results))
    assert not seen & set(session.panel.index)


def test_switch_changes_the_active_destination():
    session = make_session(999, 'structural', 'switch', 'holdout', 0)
    assert session.active == [1] * SWITCH_STEP + [0] * (TRAJECTORY_LENGTH - SWITCH_STEP)


def test_null_never_reveals_the_distinguishing_category():
    session = make_session(999, 'compositional', 'null', 'holdout', 0)
    assert all('category' not in q for q in session.queries)


def test_probes_do_not_change_model_state():
    session = make_session(999, 'structural', 'clear', 'holdout', 0)
    model = make_model('mdi', 999)
    model.observe(session.results[0], session.queries[0])
    counter = model.model._session_counter
    first = model.rank(session.panel, 10)
    second = model.rank(session.panel, 10)
    assert model.model._session_counter == counter
    assert list(first.index) == list(second.index)


def test_reference_models_rank_by_history():
    session = make_session(999, 'structural', 'clear', 'exposed', 0)
    recurrence = Recurrence()
    for rows, query in zip(session.results, session.queries):
        recurrence.observe(rows, query)
    seen = set().union(*(set(r.index) for r in session.results))
    ranked = recurrence.rank(session.panel, 5)
    if seen & set(session.panel.index):
        assert ranked.index[0] in seen
    predicate = PredicateMatch()
    predicate.observe(session.results[1], session.queries[1])
    top = predicate.rank(session.panel, 5)
    assert context_mask(top, CONTEXTS['structural'][0]).all()


def test_missing_predictions_are_misses_and_ids_are_checked():
    session = make_session(999, 'structural', 'clear', 'holdout', 0)
    _, meta = make_table(999)
    ext = set(session.panel.index[extension_mask(session.panel, 'structural', 0, meta)])
    desired = session.panel.loc[list(ext)[:3]]
    assert precision_at_budget(desired, session.panel, ext, 10) == .3
    with pytest.raises(ValueError, match='duplicate'):
        precision_at_budget(pd.concat([desired, desired]), session.panel, ext, 10)
    modified = desired.copy()
    modified.iloc[0, 0] = 'changed'
    with pytest.raises(ValueError, match='changed'):
        precision_at_budget(modified, session.panel, ext, 10)


def test_earliness_requires_staying_above_threshold():
    assert earliness([0, 0, .5, .5, .5, .5, .5, .5, .5], .3) == 2
    assert earliness([0, .5, 0, .5, .5, .5, .5, .5, .5], .3) == 3
    assert earliness([.5] * 8 + [0], .3) == TRAJECTORY_LENGTH + 1


def test_pattern_earliness_and_lag_in_summary(tmp_path):
    from query_data_predictor.destination_benchmark import summarize
    rows = []
    for seed in (1, 2):
        for target in (0, 1):
            for method, values in (('mdi', [.1, .1, .5, .9, .9, .9, .9, .9, .9]),
                                   ('random', [.1] * 9)):
                for probe, value in enumerate(values):
                    rows.append(dict(seed=seed, kind='structural', condition='clear', exposure='holdout',
                                     target=target, active=target, method=method, probe=probe, k=10,
                                     error=None, seconds=0., selected_ids=[], precision=value,
                                     other_precision=0., chance=.15, context_ceiling=.6,
                                     distinguishing_step=2))
    import query_data_predictor.destination_benchmark as db
    original = (db.write_tables, db.write_figure, db.write_paper_artifacts)
    db.write_tables = db.write_figure = db.write_paper_artifacts = lambda *a, **k: None
    try:
        summarize(tmp_path, rows)
    finally:
        db.write_tables, db.write_figure, db.write_paper_artifacts = original
    early = pd.read_csv(tmp_path / 'earliness.csv').set_index('method')
    assert early.loc['mdi', 'earliness'] == 2 and early.loc['mdi', 'lag'] == 0
    assert early.loc['mdi', 'pattern_earliness'] == 3 and early.loc['mdi', 'pattern_lag'] == 1
    assert early.loc['random', 'pattern_qualified'] == 0
