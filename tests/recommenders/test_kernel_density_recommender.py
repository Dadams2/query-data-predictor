"""
Tests for the KernelDensityRecommender class.
"""
import pandas as pd
import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock

from query_data_predictor.recommender.kernel_density_recommender import (
    KernelDensityRecommender, CandidateBudget,
)
from query_data_predictor.query_runner import QueryRunner


class TestCandidateBudget:
    """Tests for the CandidateBudget dataclass."""

    def test_initial_can_execute(self):
        budget = CandidateBudget(max_queries=3)
        assert budget.can_execute()

    def test_budget_exhausted_queries(self):
        budget = CandidateBudget(max_queries=2)
        budget.start_session()
        budget.record(10)
        budget.record(10)
        assert not budget.can_execute()

    def test_budget_exhausted_candidates(self):
        budget = CandidateBudget(max_candidates=10)
        budget.start_session()
        budget.record(15)
        assert not budget.can_execute()


class TestKernelDensityRecommender:
    """Test suite for KernelDensityRecommender."""

    @pytest.fixture
    def sample_config(self):
        return {
            'kernel_density': {
                'kernel_type': 'gaussian',
                'bandwidth_scale': 1.0,
                'temporal_decay': 0.1,
                'normalize_attributes': False,
                'candidate_sampling': {
                    'max_queries': 3,
                    'max_candidates': 500,
                    'max_time': 10.0,
                    'expansion_factor': 0.3,
                    'random_sample_limit': 50,
                },
            },
            'recommendation': {
                'top_k': 10,
            },
        }

    @pytest.fixture
    def mock_query_runner(self):
        return Mock(spec=QueryRunner)

    @pytest.fixture
    def sdss_dataframe(self):
        return pd.DataFrame({
            'specObjID': [1001, 1002, 1003],
            'z': [0.1, 0.2, 0.3],
            'ra': [180.5, 181.0, 181.5],
            'dec': [45.2, 45.5, 45.8],
        })

    def test_init_requires_query_runner(self, sample_config):
        with pytest.raises(ValueError, match="QueryRunner is required"):
            KernelDensityRecommender(sample_config)

    def test_init_with_query_runner(self, sample_config, mock_query_runner):
        rec = KernelDensityRecommender(sample_config, mock_query_runner)
        assert rec.query_runner == mock_query_runner
        assert rec.name() == "KernelDensityRecommender"

    def test_empty_input_returns_empty(self, sample_config, mock_query_runner):
        rec = KernelDensityRecommender(sample_config, mock_query_runner)
        result = rec.recommend_tuples(pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_returns_dataframe(self, sample_config, mock_query_runner, sdss_dataframe):
        """Result is always a DataFrame."""
        candidate_results = pd.DataFrame({
            'specObjID': [9001, 9002, 9003, 9004, 9005],
            'z': [0.15, 0.25, 0.35, 0.8, 0.9],
            'ra': [180.8, 181.2, 181.6, 200.0, 210.0],
            'dec': [45.3, 45.6, 45.9, 60.0, 70.0],
        })
        mock_query_runner.execute_query.return_value = candidate_results

        rec = KernelDensityRecommender(sample_config, mock_query_runner)
        result = rec.recommend_tuples(
            sdss_dataframe,
            current_query_text="SELECT specObjID, z, ra, dec FROM SpecObj WHERE z >= 0.1 and z < 0.4",
        )
        assert isinstance(result, pd.DataFrame)

    def test_sql_text_drives_extraction(self, sample_config, mock_query_runner, sdss_dataframe):
        """Providing SQL text should populate the density model with parsed regions."""
        candidate_results = pd.DataFrame({
            'specObjID': [9001],
            'z': [0.25],
            'ra': [181.0],
            'dec': [45.5],
        })
        mock_query_runner.execute_query.return_value = candidate_results

        rec = KernelDensityRecommender(sample_config, mock_query_runner)
        rec.recommend_tuples(
            sdss_dataframe,
            current_query_text="SELECT * FROM SpecObj WHERE z >= 0.1 and z < 0.4",
        )

        # Density model should have one region
        assert len(rec.density.regions) == 1
        assert "z" in rec.density.regions[0].bounds

    def test_seen_tuple_exclusion(self, sample_config, mock_query_runner, sdss_dataframe):
        """Tuples from current results should be excluded from recommendations."""
        # Return candidates that overlap with current results
        overlap_results = sdss_dataframe.copy()
        mock_query_runner.execute_query.return_value = overlap_results

        rec = KernelDensityRecommender(sample_config, mock_query_runner)
        result = rec.recommend_tuples(
            sdss_dataframe,
            current_query_text="SELECT specObjID, z, ra, dec FROM SpecObj WHERE z >= 0.1 and z < 0.4",
        )

        # All candidates were seen, so result should be empty
        assert result.empty

    def test_top_k_limiting(self, sample_config, mock_query_runner, sdss_dataframe):
        """Results are limited to top_k."""
        many_candidates = pd.DataFrame({
            'specObjID': list(range(9000, 9050)),
            'z': np.random.uniform(0.1, 0.5, 50),
            'ra': np.random.uniform(180.0, 183.0, 50),
            'dec': np.random.uniform(45.0, 46.0, 50),
        })
        mock_query_runner.execute_query.return_value = many_candidates

        rec = KernelDensityRecommender(sample_config, mock_query_runner)
        result = rec.recommend_tuples(
            sdss_dataframe,
            top_k=5,
            current_query_text="SELECT specObjID, z, ra, dec FROM SpecObj WHERE z >= 0.1 and z < 0.4",
        )
        assert len(result) <= 5

    def test_fallback_to_results_extraction(self, sample_config, mock_query_runner, sdss_dataframe):
        """When no SQL is provided, fall back to result-based extraction."""
        candidate_results = pd.DataFrame({
            'specObjID': [9001],
            'z': [0.25],
            'ra': [181.0],
            'dec': [45.5],
        })
        mock_query_runner.execute_query.return_value = candidate_results

        rec = KernelDensityRecommender(sample_config, mock_query_runner)
        # No current_query_text provided — should still work via fallback
        result = rec.recommend_tuples(sdss_dataframe)
        assert isinstance(result, pd.DataFrame)

    def test_error_handling(self, sample_config, mock_query_runner, sdss_dataframe):
        """Database errors are handled gracefully."""
        mock_query_runner.execute_query.side_effect = Exception("DB error")

        rec = KernelDensityRecommender(sample_config, mock_query_runner)
        result = rec.recommend_tuples(
            sdss_dataframe,
            current_query_text="SELECT specObjID, z, ra, dec FROM SpecObj WHERE z >= 0.1 and z < 0.4",
        )
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_multiple_invocations_accumulate_regions(self, sample_config, mock_query_runner):
        """Multiple calls to recommend_tuples accumulate regions."""
        df1 = pd.DataFrame({'specObjID': [1], 'z': [0.1], 'ra': [180.0], 'dec': [45.0]})
        df2 = pd.DataFrame({'specObjID': [2], 'z': [0.2], 'ra': [181.0], 'dec': [45.5]})
        candidate = pd.DataFrame({'specObjID': [9], 'z': [0.15], 'ra': [180.5], 'dec': [45.2]})
        mock_query_runner.execute_query.return_value = candidate

        rec = KernelDensityRecommender(sample_config, mock_query_runner)
        rec.recommend_tuples(df1, current_query_text="SELECT * FROM SpecObj WHERE z >= 0.0 and z < 0.15")
        rec.recommend_tuples(df2, current_query_text="SELECT * FROM SpecObj WHERE z >= 0.15 and z < 0.3")

        assert len(rec.density.regions) == 2

    def test_no_table_returns_empty(self, sample_config, mock_query_runner):
        """If table cannot be determined, return empty."""
        df = pd.DataFrame({'z': [0.1, 0.2]})
        rec = KernelDensityRecommender(sample_config, mock_query_runner)
        # No SQL text and DataFrame columns don't indicate a table
        result = rec.recommend_tuples(df)
        assert result.empty

    def test_name(self, sample_config, mock_query_runner):
        rec = KernelDensityRecommender(sample_config, mock_query_runner)
        assert rec.name() == "KernelDensityRecommender"

    def test_aggregation_query_preserves_schema(self, sample_config, mock_query_runner):
        """Aggregation queries (GROUP BY) should produce candidates with matching columns."""
        # Simulates a SIMBA-style aggregation query
        current_results = pd.DataFrame({
            'bin_redshift': [0.0, 1.0, 2.0],
            'bin_psfmag_u': [19.0, 20.0, 21.0],
            'count': [131, 80, 54],
        })
        # The DB query should use the same SELECT/GROUP BY, so candidates
        # should come back with the same columns
        candidate_results = pd.DataFrame({
            'bin_redshift': [3.0, 4.0],
            'bin_psfmag_u': [22.0, 23.0],
            'count': [10, 5],
        })
        mock_query_runner.execute_query.return_value = candidate_results

        rec = KernelDensityRecommender(sample_config, mock_query_runner)
        sql = ("SELECT FLOOR(redshift/1.0) AS bin_redshift, "
               "FLOOR(psfMag_u/1) AS bin_psfMag_u, "
               "COUNT(*) as count FROM sdss_photobj "
               "GROUP BY 1, bin_redshift, bin_psfMag_u")

        result = rec.recommend_tuples(current_results, current_query_text=sql)
        assert isinstance(result, pd.DataFrame)

        # Check that the generated SQL preserved the aggregation structure
        call_args = mock_query_runner.execute_query.call_args_list
        for call in call_args:
            query = call[0][0]
            assert 'GROUP BY' in query.upper()
            assert 'FLOOR(' in query.upper() or 'COUNT(' in query.upper()

    def test_parse_query_skeleton(self, sample_config, mock_query_runner):
        """_parse_query_skeleton correctly extracts SELECT, FROM, GROUP BY."""
        rec = KernelDensityRecommender(sample_config, mock_query_runner)
        sql = ("SELECT FLOOR(redshift/1.0) AS bin_redshift, COUNT(*) as count "
               "FROM sdss_photobj GROUP BY 1, bin_redshift")
        skeleton = rec._parse_query_skeleton(sql)
        assert skeleton is not None
        assert skeleton['from_table'] == 'sdss_photobj'
        assert skeleton['is_aggregation'] is True
        assert 'FLOOR' in skeleton['select']
        assert 'GROUP BY' not in skeleton['select']

    def test_parse_query_skeleton_no_group_by(self, sample_config, mock_query_runner):
        """Non-aggregation queries return is_aggregation=False."""
        rec = KernelDensityRecommender(sample_config, mock_query_runner)
        sql = "SELECT z, ra, dec FROM SpecObj WHERE z > 0.1"
        skeleton = rec._parse_query_skeleton(sql)
        assert skeleton is not None
        assert skeleton['is_aggregation'] is False
        assert skeleton['group_by'] is None

    def test_get_real_attr_name_maps_bin_columns(self, sample_config, mock_query_runner):
        """_get_real_attr_name maps bin_redshift -> redshift via bin_columns."""
        from query_data_predictor.recommender.region_extractor import QueryRegion
        rec = KernelDensityRecommender(sample_config, mock_query_runner)
        region = QueryRegion(bin_columns={'redshift': 1.0, 'psfMag_u': 1.0})
        assert rec._get_real_attr_name('bin_redshift', region) == 'redshift'
        assert rec._get_real_attr_name('bin_psfmag_u', region) == 'psfMag_u'
        assert rec._get_real_attr_name('count', region) is None
        assert rec._get_real_attr_name('ra', region) == 'ra'
