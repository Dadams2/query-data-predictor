"""
Tests for the InterestDensity class.
"""
import numpy as np
import pytest

from query_data_predictor.recommender.region_extractor import (
    QueryRegion, AttributeBound,
)
from query_data_predictor.recommender.interest_density import (
    InterestDensity, KernelConfig,
)


def _make_region(bounds_dict=None, table_name="TestTable"):
    """Helper to build a QueryRegion from a simple dict of {attr: (lo, hi)}."""
    bounds = {}
    if bounds_dict:
        for attr, (lo, hi) in bounds_dict.items():
            bounds[attr] = AttributeBound(intervals=[(lo, hi)])
    return QueryRegion(table_name=table_name, bounds=bounds)


class TestIndicatorKernel:
    """Tests for the indicator kernel."""

    @pytest.fixture
    def density(self):
        config = KernelConfig(kernel_type="indicator", temporal_decay=0.0)
        return InterestDensity(config)

    def test_inside_region(self, density):
        region = _make_region({"z": (0.1, 0.5), "ra": (180.0, 200.0)})
        density.add_region(region)

        score = density.score_point({"z": 0.3, "ra": 190.0})
        assert score == pytest.approx(1.0)

    def test_outside_region(self, density):
        region = _make_region({"z": (0.1, 0.5)})
        density.add_region(region)

        score = density.score_point({"z": 0.8})
        assert score == pytest.approx(0.0)

    def test_on_boundary(self, density):
        region = _make_region({"z": (0.1, 0.5)})
        density.add_region(region)

        # Both endpoints should be inside (inclusive)
        assert density.score_point({"z": 0.1}) == pytest.approx(1.0)
        assert density.score_point({"z": 0.5}) == pytest.approx(1.0)

    def test_multiple_regions(self, density):
        r1 = _make_region({"z": (0.1, 0.3)})
        r2 = _make_region({"z": (0.4, 0.6)})
        density.add_region(r1)
        density.add_region(r2)

        # Point inside r1 only
        score = density.score_point({"z": 0.2})
        assert score == pytest.approx(0.5)  # 1 hit out of 2 regions

        # Point inside r2 only
        score = density.score_point({"z": 0.5})
        assert score == pytest.approx(0.5)

        # Point outside both
        score = density.score_point({"z": 0.35})
        assert score == pytest.approx(0.0)


class TestGaussianKernel:
    """Tests for the Gaussian kernel."""

    @pytest.fixture
    def density(self):
        config = KernelConfig(kernel_type="gaussian", temporal_decay=0.0, bandwidth_scale=1.0, normalize_attributes=False)
        return InterestDensity(config)

    def test_centroid_highest_score(self, density):
        region = _make_region({"z": (0.0, 1.0)})
        density.add_region(region)

        centroid_score = density.score_point({"z": 0.5})
        off_center_score = density.score_point({"z": 0.8})
        far_score = density.score_point({"z": 5.0})

        assert centroid_score > off_center_score
        assert off_center_score > far_score

    def test_monotonic_falloff(self, density):
        region = _make_region({"z": (0.0, 1.0)})
        density.add_region(region)

        distances = [0.0, 0.5, 1.0, 2.0, 5.0]
        centroid = 0.5
        scores = [density.score_point({"z": centroid + d}) for d in distances]

        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_centroid_score_is_one(self, density):
        """At the centroid, exp(0) = 1."""
        region = _make_region({"z": (0.0, 1.0)})
        density.add_region(region)

        score = density.score_point({"z": 0.5})
        assert score == pytest.approx(1.0)

    def test_multidimensional_gaussian(self, density):
        region = _make_region({"z": (0.0, 1.0), "ra": (180.0, 200.0)})
        density.add_region(region)

        centroid_score = density.score_point({"z": 0.5, "ra": 190.0})
        off_score = density.score_point({"z": 0.5, "ra": 210.0})

        assert centroid_score > off_score


class TestRegionAdaptedKernel:
    """Tests for the region-adapted kernel."""

    @pytest.fixture
    def density(self):
        config = KernelConfig(kernel_type="region_adapted", temporal_decay=0.0, bandwidth_scale=1.0)
        return InterestDensity(config)

    def test_centroid_highest(self, density):
        region = _make_region({"z": (0.0, 1.0)})
        density.add_region(region)

        centroid_score = density.score_point({"z": 0.5})
        far_score = density.score_point({"z": 5.0})
        assert centroid_score > far_score

    def test_wider_region_wider_kernel(self):
        """A wider region should give a broader kernel (higher score at same distance)."""
        narrow_config = KernelConfig(kernel_type="region_adapted", temporal_decay=0.0, bandwidth_scale=1.0)
        narrow = InterestDensity(narrow_config)
        narrow.add_region(_make_region({"z": (0.0, 0.1)}))

        wide_config = KernelConfig(kernel_type="region_adapted", temporal_decay=0.0, bandwidth_scale=1.0)
        wide = InterestDensity(wide_config)
        wide.add_region(_make_region({"z": (0.0, 10.0)}))

        test_point = {"z": 2.0}
        narrow_score = narrow.score_point(test_point)
        wide_score = wide.score_point(test_point)

        assert wide_score > narrow_score


class TestTemporalDecay:
    """Tests for temporal decay weighting."""

    def test_recent_region_weighted_higher(self):
        config = KernelConfig(kernel_type="indicator", temporal_decay=0.5)
        density = InterestDensity(config)

        old_region = _make_region({"z": (0.0, 0.5)})
        new_region = _make_region({"z": (0.4, 0.9)})

        density.add_region(old_region)
        density.add_region(new_region)

        # Point in new region only
        score_new = density.score_point({"z": 0.7})
        # Point in old region only
        score_old = density.score_point({"z": 0.1})

        # New region contribution should be higher due to recency
        assert score_new > score_old

    def test_no_decay_equal_weight(self):
        config = KernelConfig(kernel_type="indicator", temporal_decay=0.0)
        density = InterestDensity(config)

        r1 = _make_region({"z": (0.0, 0.5)})
        r2 = _make_region({"z": (1.0, 1.5)})

        density.add_region(r1)
        density.add_region(r2)

        score_r1 = density.score_point({"z": 0.2})
        score_r2 = density.score_point({"z": 1.2})

        assert score_r1 == pytest.approx(score_r2)


class TestBatchScoring:
    """Tests for batch scoring consistency."""

    def test_batch_matches_individual(self):
        config = KernelConfig(kernel_type="gaussian", temporal_decay=0.1, normalize_attributes=False)
        density = InterestDensity(config)
        density.add_region(_make_region({"z": (0.0, 1.0)}))
        density.add_region(_make_region({"z": (0.5, 1.5)}))

        points = [{"z": 0.3}, {"z": 0.7}, {"z": 1.2}, {"z": 5.0}]
        batch_scores = density.score_points_batch(points)
        individual_scores = [density.score_point(p) for p in points]

        np.testing.assert_allclose(batch_scores, individual_scores)

    def test_empty_batch(self):
        config = KernelConfig(kernel_type="gaussian")
        density = InterestDensity(config)
        density.add_region(_make_region({"z": (0.0, 1.0)}))

        scores = density.score_points_batch([])
        assert len(scores) == 0


class TestEmptyRegionHandling:
    """Tests for empty region (no WHERE clause) handling."""

    def test_empty_region_base_score(self):
        config = KernelConfig(kernel_type="gaussian", temporal_decay=0.0, normalize_attributes=False)
        density = InterestDensity(config)
        density.add_region(QueryRegion(table_name="SpecObj"))  # No bounds

        score = density.score_point({"z": 0.5})
        assert score == pytest.approx(InterestDensity.BASE_SCORE)

    def test_no_regions(self):
        config = KernelConfig(kernel_type="gaussian")
        density = InterestDensity(config)

        score = density.score_point({"z": 0.5})
        assert score == 0.0


class TestHighDensityBounds:
    """Tests for get_high_density_bounds."""

    def test_single_region(self):
        config = KernelConfig()
        density = InterestDensity(config)
        density.add_region(_make_region({"z": (0.1, 0.5)}))

        bounds = density.get_high_density_bounds()
        assert "z" in bounds
        assert bounds["z"] == (0.1, 0.5)

    def test_multiple_regions_combined(self):
        config = KernelConfig()
        density = InterestDensity(config)
        density.add_region(_make_region({"z": (0.1, 0.5)}))
        density.add_region(_make_region({"z": (0.3, 0.8)}))

        bounds = density.get_high_density_bounds()
        assert bounds["z"] == (0.1, 0.8)

    def test_no_regions(self):
        config = KernelConfig()
        density = InterestDensity(config)
        assert density.get_high_density_bounds() == {}
