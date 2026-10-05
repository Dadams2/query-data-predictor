import pandas as pd
import pytest

from query_data_predictor.destination_benchmark import (
    CONDITIONS, CONTEXTS, KINDS, TRAJECTORY_LENGTH, SWITCH_STEP, PredicateMatch, Recurrence,
    context_mask, earliness, extension_mask, make_model, make_session, make_table, precision_at_budget,
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
    # Contexts are disjoint, so extensions never overlap.
    assert not (extension_mask(table, kind, 0, meta) & extension_mask(table, kind, 1, meta)).any()


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
