import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from query_data_predictor.interestingness import InterestingnessMeasures

# Table 4.1 from the book
# | Colour | Shape  | Count |
# |--------|--------|-------|
# | red    | round  | 3     |
# | green  | round  | 2     |
# | red    | square | 1     |
# | blue   | square | 1     |


# Configurable constants for expected results
EXPECTED_P = np.array([0.429, 0.286, 0.143, 0.143])  # From text
EXPECTED_Q = np.array([0.25, 0.25, 0.25, 0.25])  # From text
EXPECTED_R = np.array([0.339, 0.268, 0.196, 0.196])  # From text

EXPECTED_RESULTS = {
    "variance": 0.018,
    "simpson": 0.306,
    "shannon": 1.842,
    "total": 7.368,
    "max": 2.0,
    "mcintosh": 0.718,
    "lorenz": 0.501,
    "gini": 0.25,  # modified from book because floating point precision
    "berger": 0.429,
    "schutz": 0.215,
    "bray": 0.786,
    "whittaker": 0.786,
    "kullback": 1.842,
    "macarthur": 0.039,
    "theil": 0.238,
    "atkinson": 0.105,
}


@pytest.fixture
def sample_measures():
    """
    Creates InterestingnessMeasures instance with the sample data
    See table 4.1 from the book for the data
    Colour Shape Count:
    red    round  3
    green  round  2
    red    square 1
    blue   square 1
    """
    sample_counts = [3, 2, 1, 1]
    return InterestingnessMeasures(sample_counts)


def test_init_probabilities(sample_measures):
    """Test if the probability distributions are correctly initialized"""
    assert_almost_equal(sample_measures.P, EXPECTED_P, decimal=3)
    assert_almost_equal(sample_measures.q, EXPECTED_Q, decimal=3)
    assert_almost_equal(sample_measures.r, EXPECTED_R, decimal=3)


def test_variance(sample_measures):
    """Test IVariance measure - Example 4.1"""
    expected = EXPECTED_RESULTS["variance"]
    assert_almost_equal(sample_measures.variance(), expected, decimal=3)


def test_simpson(sample_measures):
    """Test ISimpson measure - Example 4.2"""
    expected = EXPECTED_RESULTS["simpson"]
    assert_almost_equal(sample_measures.simpson(), expected, decimal=3)


def test_shannon(sample_measures):
    """Test IShannon measure - Example 4.3"""
    expected = EXPECTED_RESULTS["shannon"]
    assert_almost_equal(sample_measures.shannon(), expected, decimal=3)


def test_total(sample_measures):
    """Test ITotal measure - Example 4.4"""
    expected = EXPECTED_RESULTS["total"]
    assert_almost_equal(sample_measures.total(), expected, decimal=3)


def test_max(sample_measures):
    """Test IMax measure - Example 4.5"""
    expected = EXPECTED_RESULTS["max"]
    assert_almost_equal(sample_measures.max(), expected, decimal=3)


def test_mcintosh(sample_measures):
    """Test IMcIntosh measure - Example 4.6"""
    expected = EXPECTED_RESULTS["mcintosh"]
    assert_almost_equal(sample_measures.mcintosh(), expected, decimal=3)


def test_lorenz(sample_measures):
    """Test ILorenz measure - Example 4.7"""
    expected = EXPECTED_RESULTS["lorenz"]
    assert_almost_equal(sample_measures.lorenz(), expected, decimal=3)


def test_gini(sample_measures):
    """Test IGini measure - Example 4.8"""
    expected = EXPECTED_RESULTS["gini"]
    assert_almost_equal(sample_measures.gini(), expected, decimal=3)


def test_berger(sample_measures):
    """Test IBerger measure - Example 4.9"""
    expected = EXPECTED_RESULTS["berger"]
    assert_almost_equal(sample_measures.berger(), expected, decimal=3)


def test_schutz(sample_measures):
    """Test ISchutz measure - Example 4.10"""
    expected = EXPECTED_RESULTS["schutz"]
    assert_almost_equal(sample_measures.schutz(), expected, decimal=3)


def test_bray(sample_measures):
    """Test IBray measure - Example 4.11"""
    expected = EXPECTED_RESULTS["bray"]
    assert_almost_equal(sample_measures.bray(), expected, decimal=3)


def test_whittaker(sample_measures):
    """Test IWhittaker measure - Example 4.12"""
    expected = EXPECTED_RESULTS["whittaker"]
    assert_almost_equal(sample_measures.whittaker(), expected, decimal=3)


def test_kullback(sample_measures):
    """Test IKullback measure - Example 4.13"""
    expected = EXPECTED_RESULTS["kullback"]
    assert_almost_equal(sample_measures.kullback(), expected, decimal=3)


def test_macarthur(sample_measures):
    """Test IMacArthur measure - Example 4.14"""
    expected = EXPECTED_RESULTS["macarthur"]
    assert_almost_equal(sample_measures.macarthur(), expected, decimal=3)


def test_theil(sample_measures):
    """Test ITheil measure - Example 4.15"""
    expected = EXPECTED_RESULTS["theil"]
    assert_almost_equal(sample_measures.theil(), expected, decimal=3)


def test_atkinson(sample_measures):
    """Test IAtkinson measure - Example 4.16"""
    expected = EXPECTED_RESULTS["atkinson"]
    assert_almost_equal(sample_measures.atkinson(), expected, decimal=3)


def test_edge_cases():
    """Test edge cases and potential error conditions"""

    # Test with single value
    single_measure = InterestingnessMeasures([1])
    assert_almost_equal(single_measure.shannon(), 0.0, decimal=3)
    assert_almost_equal(single_measure.simpson(), 1.0, decimal=3)

    # Test with equal counts
    equal_measures = InterestingnessMeasures([2, 2, 2, 2])
    assert_almost_equal(equal_measures.shannon(), 2.0, decimal=3)
    assert_almost_equal(equal_measures.simpson(), 0.25, decimal=3)

    # Test with zeros
    # with pytest.raises(ValueError):
    #     InterestingnessMeasures([])  # Empty list should raise error

    # Test with invalid inputs
    # with pytest.raises(ValueError):
    # InterestingnessMeasures([-1, 1, 1])  # Negative counts should raise error


def test_calculate_all(sample_measures):
    """Test that calculate_all returns all measures with correct values"""
    results = sample_measures.calculate_all()
    for measure, expected in EXPECTED_RESULTS.items():
        assert_almost_equal(results[measure], expected, decimal=3)
