"""
Tests for the RegionExtractor class.
"""
import pandas as pd
import pytest

from query_data_predictor.recommender.region_extractor import (
    RegionExtractor, QueryRegion, AttributeBound,
)


class TestAttributeBound:
    """Tests for the AttributeBound dataclass."""

    def test_centroid_single_interval(self):
        bound = AttributeBound(intervals=[(10.0, 20.0)])
        assert bound.centroid == 15.0

    def test_centroid_multiple_intervals(self):
        bound = AttributeBound(intervals=[(0.0, 10.0), (20.0, 30.0)])
        assert bound.centroid == 15.0  # (0 + 30) / 2

    def test_centroid_empty(self):
        bound = AttributeBound()
        assert bound.centroid is None

    def test_extent_single_interval(self):
        bound = AttributeBound(intervals=[(10.0, 20.0)])
        assert bound.extent == 10.0

    def test_extent_multiple_intervals(self):
        bound = AttributeBound(intervals=[(0.0, 10.0), (20.0, 30.0)])
        assert bound.extent == 30.0  # 30 - 0

    def test_extent_zero_width(self):
        bound = AttributeBound(intervals=[(5.0, 5.0)])
        assert bound.extent == pytest.approx(1e-10)

    def test_overall_min_max(self):
        bound = AttributeBound(intervals=[(5.0, 10.0), (20.0, 30.0)])
        assert bound.overall_min == 5.0
        assert bound.overall_max == 30.0

    def test_is_numeric_and_categorical(self):
        numeric = AttributeBound(intervals=[(1.0, 2.0)])
        assert numeric.is_numeric
        assert not numeric.is_categorical

        categorical = AttributeBound(categorical_values={"A", "B"})
        assert not categorical.is_numeric
        assert categorical.is_categorical

        both = AttributeBound(intervals=[(1.0, 2.0)], categorical_values={"A"})
        assert both.is_numeric
        assert both.is_categorical


class TestRegionExtractor:
    """Tests for the RegionExtractor class."""

    @pytest.fixture
    def extractor(self):
        return RegionExtractor()

    # --- SIMBA range predicates ---

    def test_single_attribute_or_ranges(self, extractor):
        """SIMBA pattern: single attribute with OR-connected ranges."""
        sql = """
        SELECT FLOOR(obsdate/15.0) AS bin_obsdate, count(*) AS cnt
        FROM SpecObj
        WHERE (obsdate >= 20130915.0 and obsdate < 20130930.0)
           or (obsdate >= 20131001.0 and obsdate < 20131015.0)
        GROUP BY 1 ORDER BY 1
        """
        region = extractor.extract(sql)

        assert region.table_name == "SpecObj"
        assert "obsdate" in region.bounds
        bound = region.bounds["obsdate"]
        assert len(bound.intervals) == 2
        assert bound.intervals[0] == (20130915.0, 20130930.0)
        assert bound.intervals[1] == (20131001.0, 20131015.0)
        assert bound.overall_min == 20130915.0
        assert bound.overall_max == 20131015.0

    def test_multi_attribute_drilldown(self, extractor):
        """Multi-attribute with AND between groups, OR within groups."""
        sql = """
        SELECT z, ra FROM SpecObj
        WHERE (z >= 0.1 and z < 0.3) or (z >= 0.5 and z < 0.8)
          AND (ra >= 180.0 and ra < 200.0)
        """
        region = extractor.extract(sql)

        assert "z" in region.bounds
        assert len(region.bounds["z"].intervals) == 2
        assert "ra" in region.bounds
        assert len(region.bounds["ra"].intervals) >= 1

    def test_categorical_equality(self, extractor):
        """Categorical equality predicates."""
        sql = "SELECT * FROM SpecObj WHERE SpecClass = 'QSO' or SpecClass = 'GALAXY'"
        region = extractor.extract(sql)

        assert "SpecClass" in region.bounds
        bound = region.bounds["SpecClass"]
        assert bound.is_categorical
        assert "QSO" in bound.categorical_values
        assert "GALAXY" in bound.categorical_values

    def test_combined_range_and_categorical(self, extractor):
        """Combined range and categorical predicates."""
        sql = """
        SELECT * FROM SpecObj
        WHERE (z >= 0.1 and z < 0.5) AND SpecClass = 'QSO'
        """
        region = extractor.extract(sql)

        assert "z" in region.bounds
        assert region.bounds["z"].is_numeric
        assert "SpecClass" in region.bounds
        assert region.bounds["SpecClass"].is_categorical

    def test_between_predicate(self, extractor):
        """Standard BETWEEN predicate."""
        sql = "SELECT * FROM SpecObj WHERE z BETWEEN 0.1 AND 0.5"
        region = extractor.extract(sql)

        assert "z" in region.bounds
        assert region.bounds["z"].intervals[0] == (0.1, 0.5)

    def test_simple_greater_than(self, extractor):
        """Simple > comparison."""
        sql = "SELECT * FROM SpecObj WHERE z > 0.5"
        region = extractor.extract(sql)

        assert "z" in region.bounds
        lo, hi = region.bounds["z"].intervals[0]
        assert lo == 0.5
        assert hi > 1e9  # large upper bound

    def test_simple_less_than(self, extractor):
        """Simple < comparison."""
        sql = "SELECT * FROM SpecObj WHERE ra < 200.0"
        region = extractor.extract(sql)

        assert "ra" in region.bounds
        lo, hi = region.bounds["ra"].intervals[0]
        assert lo < -1e9  # large lower bound
        assert hi == 200.0

    def test_no_where_clause(self, extractor):
        """No WHERE clause returns empty bounds."""
        sql = "SELECT * FROM SpecObj"
        region = extractor.extract(sql)

        assert region.table_name == "SpecObj"
        assert region.bounds == {}

    def test_floor_bin_extraction(self, extractor):
        """FLOOR(attr/divisor) AS bin_attr pattern."""
        sql = "SELECT FLOOR(obsdate/15.0) AS bin_obsdate, FLOOR(z/0.01) AS bin_z FROM SpecObj"
        region = extractor.extract(sql)

        assert "obsdate" in region.bin_columns
        assert region.bin_columns["obsdate"] == 15.0
        assert "z" in region.bin_columns
        assert region.bin_columns["z"] == 0.01

    def test_table_name_extraction(self, extractor):
        """FROM clause table name extraction."""
        sql = "SELECT a, b FROM MyTable WHERE a > 1"
        region = extractor.extract(sql)
        assert region.table_name == "MyTable"

    def test_select_columns_extraction(self, extractor):
        """SELECT clause column extraction."""
        sql = "SELECT z, ra, dec FROM SpecObj WHERE z > 0.1"
        region = extractor.extract(sql)
        assert region.select_columns == ["z", "ra", "dec"]

    def test_select_star(self, extractor):
        """SELECT * returns empty select_columns."""
        sql = "SELECT * FROM SpecObj"
        region = extractor.extract(sql)
        assert region.select_columns == []

    def test_raw_sql_preserved(self, extractor):
        """Raw SQL is preserved in the region."""
        sql = "SELECT * FROM SpecObj WHERE z > 0.1"
        region = extractor.extract(sql)
        assert region.raw_sql == sql


class TestExtractFromResults:
    """Tests for the fallback DataFrame-based extraction."""

    @pytest.fixture
    def extractor(self):
        return RegionExtractor()

    def test_numeric_bounds_from_dataframe(self, extractor):
        """Infer bounds from numeric DataFrame columns."""
        df = pd.DataFrame({
            'z': [0.1, 0.2, 0.3, 0.4],
            'ra': [180.0, 181.0, 182.0, 183.0],
        })
        region = extractor.extract_from_results(df)

        assert "z" in region.bounds
        assert region.bounds["z"].overall_min == pytest.approx(0.1)
        assert region.bounds["z"].overall_max == pytest.approx(0.4)
        assert "ra" in region.bounds

    def test_categorical_from_dataframe(self, extractor):
        """Infer categorical values from string DataFrame columns."""
        df = pd.DataFrame({
            'z': [0.1, 0.2],
            'SpecClass': ['QSO', 'GALAXY'],
        })
        region = extractor.extract_from_results(df)

        assert "SpecClass" in region.bounds
        assert region.bounds["SpecClass"].is_categorical
        assert "QSO" in region.bounds["SpecClass"].categorical_values

    def test_empty_dataframe(self, extractor):
        """Empty DataFrame returns region with no bounds."""
        df = pd.DataFrame()
        region = extractor.extract_from_results(df)
        assert region.bounds == {}

    def test_sql_parsed_when_provided(self, extractor):
        """SQL string is also parsed when provided alongside DataFrame."""
        df = pd.DataFrame({'z': [0.1, 0.2]})
        sql = "SELECT z FROM SpecObj"
        region = extractor.extract_from_results(df, sql=sql)

        assert region.table_name == "SpecObj"
        assert "z" in region.bounds

    def test_select_columns_from_dataframe(self, extractor):
        """Columns come from DataFrame when not extracted from SQL."""
        df = pd.DataFrame({'z': [0.1], 'ra': [180.0]})
        region = extractor.extract_from_results(df)
        assert set(region.select_columns) == {"z", "ra"}
