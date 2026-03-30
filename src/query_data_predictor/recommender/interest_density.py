"""
Interest Density - Kernel density estimation over query regions.

Estimates an interest density function from the sequence of SQL query regions,
then scores candidate tuples by their estimated interest. Supports indicator,
Gaussian, and region-adapted kernels with temporal decay weighting.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import logging

from .region_extractor import QueryRegion, AttributeBound

logger = logging.getLogger(__name__)


@dataclass
class KernelConfig:
    """Configuration for the kernel density estimator."""
    kernel_type: str = "gaussian"  # "indicator", "gaussian", "region_adapted"
    bandwidth_scale: float = 1.0
    temporal_decay: float = 0.1
    normalize_attributes: bool = True


class InterestDensity:
    """Kernel density estimation engine over query regions.

    Maintains a history of query regions and computes an interest density
    function f(x) = (1/t) * Σ wᵢ * K(x, Rᵢ) where wᵢ are temporal weights
    and K is a kernel function.
    """

    BASE_SCORE = 0.1  # Score contribution from empty regions (no WHERE clause)

    def __init__(self, config: KernelConfig = None):
        self.config = config or KernelConfig()
        self.regions: List[QueryRegion] = []
        self._attribute_stats: Dict[str, Tuple[float, float]] = {}  # attr -> (mean, std)

    def add_region(self, region: QueryRegion):
        """Append a query region to the history."""
        self.regions.append(region)

    def update_attribute_stats(self, stats: Dict[str, Tuple[float, float]]):
        """Set normalization parameters (mean, std) per attribute."""
        self._attribute_stats.update(stats)

    def score_point(self, x: Dict[str, float]) -> float:
        """Compute the interest density f(x) at a single point.

        Args:
            x: Dictionary mapping attribute names to values

        Returns:
            Estimated interest density score (non-negative float)
        """
        if not self.regions:
            return 0.0

        t = len(self.regions)
        total = 0.0

        for i, region in enumerate(self.regions):
            w = self._temporal_weight(i)
            k = self._evaluate_kernel(x, region)
            total += w * k

        return total / t

    def score_points_batch(self, points: List[Dict[str, float]]) -> np.ndarray:
        """Vectorized scoring of multiple points.

        Args:
            points: List of dictionaries mapping attribute names to values

        Returns:
            Array of interest density scores
        """
        if not points:
            return np.array([])

        scores = np.array([self.score_point(p) for p in points])
        return scores

    def get_high_density_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Compute bounds that cover the high-density region.

        Returns the overall min/max across all region bounds per attribute,
        useful for constructing candidate sampling queries.
        """
        combined: Dict[str, Tuple[float, float]] = {}

        for region in self.regions:
            for attr, bound in region.bounds.items():
                if bound.overall_min is not None and bound.overall_max is not None:
                    if attr in combined:
                        old_lo, old_hi = combined[attr]
                        combined[attr] = (
                            min(old_lo, bound.overall_min),
                            max(old_hi, bound.overall_max),
                        )
                    else:
                        combined[attr] = (bound.overall_min, bound.overall_max)

        return combined

    def _temporal_weight(self, i: int) -> float:
        """Compute temporal weight for region at index i.

        More recent regions get higher weight:
        w(i) = exp(-λ * (t - i - 1))
        """
        t = len(self.regions)
        decay = self.config.temporal_decay
        return np.exp(-decay * (t - i - 1))

    def _evaluate_kernel(self, x: Dict[str, float], region: QueryRegion) -> float:
        """Evaluate the kernel function K(x, R)."""
        if not region.bounds:
            return self.BASE_SCORE

        kernel_type = self.config.kernel_type
        if kernel_type == "indicator":
            return self._kernel_indicator(x, region)
        elif kernel_type == "gaussian":
            return self._kernel_gaussian(x, region)
        elif kernel_type == "region_adapted":
            return self._kernel_region_adapted(x, region)
        else:
            logger.warning(f"Unknown kernel type '{kernel_type}', falling back to gaussian")
            return self._kernel_gaussian(x, region)

    def _kernel_indicator(self, x: Dict[str, float], region: QueryRegion) -> float:
        """Indicator kernel: 1 if x is inside region on all bounded attributes, else 0."""
        for attr, bound in region.bounds.items():
            if attr not in x:
                continue

            val = x[attr]

            if bound.is_numeric:
                inside = any(lo <= val <= hi for lo, hi in bound.intervals)
                if not inside:
                    return 0.0

            if bound.is_categorical:
                if str(val) not in bound.categorical_values:
                    return 0.0

        return 1.0

    def _kernel_gaussian(self, x: Dict[str, float], region: QueryRegion) -> float:
        """Gaussian kernel: exp(-||x - centroid(R)||² / (2σ²)).

        σ per attribute = extent * bandwidth_scale.
        Only uses attributes present in both x and region bounds.
        """
        sq_dist = 0.0
        n_attrs = 0

        for attr, bound in region.bounds.items():
            if attr not in x or not bound.is_numeric:
                continue

            centroid = bound.centroid
            if centroid is None:
                continue

            extent = bound.extent
            sigma = extent * self.config.bandwidth_scale
            if sigma < 1e-20:
                sigma = 1e-10

            val = x[attr]
            diff = val - centroid

            # Apply normalization if available
            if self.config.normalize_attributes and attr in self._attribute_stats:
                mean, std = self._attribute_stats[attr]
                if std > 1e-20:
                    diff = (val - mean) / std - (centroid - mean) / std
                    sigma = sigma / std

            sq_dist += (diff / sigma) ** 2
            n_attrs += 1

        if n_attrs == 0:
            return self.BASE_SCORE

        return np.exp(-sq_dist / 2.0)

    def _kernel_region_adapted(self, x: Dict[str, float], region: QueryRegion) -> float:
        """Region-adapted kernel using Mahalanobis distance with diagonal covariance.

        Uses region extents as the diagonal of the covariance matrix.
        """
        mahal_sq = 0.0
        n_attrs = 0

        for attr, bound in region.bounds.items():
            if attr not in x or not bound.is_numeric:
                continue

            centroid = bound.centroid
            extent = bound.extent
            if centroid is None or extent is None:
                continue

            val = x[attr]
            variance = (extent * self.config.bandwidth_scale) ** 2
            if variance < 1e-30:
                variance = 1e-20

            mahal_sq += (val - centroid) ** 2 / variance
            n_attrs += 1

        if n_attrs == 0:
            return self.BASE_SCORE

        # Use multivariate Gaussian-like kernel with Mahalanobis distance
        return np.exp(-mahal_sq / 2.0)
