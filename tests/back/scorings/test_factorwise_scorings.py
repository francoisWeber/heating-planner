import pandas as pd
import geopandas as gpd
import numpy as np
from heating_planner.back.scoring.factorwise_scoring import lower_better_score, higher_better_score, neutral_score, FactorwiseScoring
from heating_planner.back.data.base import Factor, FactorTrend, FactorType
import pytest
from shapely.geometry import Point


@pytest.fixture
def df():
    return pd.DataFrame(
        data={
            "f1": [1, 2, 3, 4],
            "f2": [10, 20, 30, 40],
            "f3": [11, 22, 33, 44],
        },
        dtype=float,
    )


@pytest.fixture
def factors():
    return [
        Factor(name="f1", description="Factor 1", trend=FactorTrend.LOWER_BETTER, type=FactorType.CONTINUOUS),
        Factor(name="f2", description="Factor 2", trend=FactorTrend.HIGHER_BETTER, type=FactorType.CONTINUOUS),
        Factor(name="f3", description="Factor 3", trend=FactorTrend.NEUTRAL, type=FactorType.CONTINUOUS),
    ]


@pytest.fixture
def optimal_ranges():
    return {"f1": [2, 3], "f2": [5, 25], "f3": [30, 35]}


def test_lower_better_score():
    x = np.linspace(0, 10, 11)
    lower_bound = 2
    upper_bound = 4
    score = lower_better_score(x, lower_bound, upper_bound)
    expected = np.array([0.2, 0.1, 0, 0, 0, 1, 2, 3, 4, 5, 6])
    assert np.allclose(score, expected)


def test_higher_better_score():
    x = np.linspace(0, 10, 11)
    lower_bound = 2
    upper_bound = 4
    score = higher_better_score(x, lower_bound, upper_bound)
    expected = np.array([2, 1, 0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert np.allclose(score, expected)


def test_neutral_score():
    x = np.linspace(0, 10, 11)
    lower_bound = 2
    upper_bound = 4
    score = neutral_score(x, lower_bound, upper_bound)
    expected = np.array([2, 1, 0, 0, 0, 1, 2, 3, 4, 5, 6])
    assert np.allclose(score, expected)


def test_factorwise_scoring_by_type(df, factors):
    scores = FactorwiseScoring._by_trend(df, factors)
    expected = pd.DataFrame(
        {
            "f1": [-1, -2, -3, -4],
            "f2": [10, 20, 30, 40],
        },
        dtype=float,
    )
    pd.testing.assert_frame_equal(scores, expected)


def test_factorwise_scoring_by_factor_optimal_range_discrepancy(df, factors, optimal_ranges):
    scores = FactorwiseScoring._by_optimal_range_discrepancy(df, factors, optimal_ranges)
    expected = pd.DataFrame(
        {
            "f1": [0.1, 0, 0, 1],
            "f2": [0, 0, 0.5, 1.5],
            "f3": [19, 8, 0, 9],
        },
        dtype=float,
    )
    pd.testing.assert_frame_equal(scores, expected)


def test_factorwise_scoring_by_factor_reference_values(df, factors):
    geometry = [Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)]
    df_ref = gpd.GeoDataFrame(
        data={
            "f1": [2, 1, 3, 8],  # [1, 2, 3, 4] / LOWER
            "f2": [5, 40, 30, 40],  # [10, 20, 30, 40] / HIGHER
            "fX": [11, 22, 33, 44],
        },
        geometry=geometry,
        dtype=float,
    )
    df_proj = gpd.GeoDataFrame(df, geometry=geometry)
    scores = FactorwiseScoring._by_historical_value(df_ref, df_proj, factors)
    expected = pd.DataFrame(
        {
            "f1": [-1 * (1 - 2) / 2, -1 * (2 - 1) / 1, -1 * (3 - 3) / 3, -1 * (4 - 8) / 8],
            "f2": [(10 - 5) / 5, (20 - 40) / 40, 0, 0],
        },
        dtype=float,
    )
    pd.testing.assert_frame_equal(scores, expected)
