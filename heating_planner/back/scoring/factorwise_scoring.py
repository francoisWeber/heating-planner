from typing import Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd

from heating_planner.back.data.base import HazardDataset
from heating_planner.back.data.model.factor import Factor, FactorTrend, FactorType
from heating_planner.back.streamlit_enums import StreamlitReadyEnum
from heating_planner.back.scoring.scaler import GeoPandasScalingStrategy


SCORE_COL = "score"
GOOD_SIDE_COEF = 0.1
BAD_SIDE_COEF = 1


def lower_better_score(x: np.ndarray, lower_bound: float, upper_bound: float) -> np.ndarray:
    score = np.zeros_like(x, dtype=float)
    score += np.where(x < lower_bound, (lower_bound - x) * GOOD_SIDE_COEF, 0)
    score += np.where(x > upper_bound, (x - upper_bound) * BAD_SIDE_COEF, 0)
    return -1 * score


def higher_better_score(x: np.ndarray, lower_bound: float, upper_bound: float) -> np.ndarray:
    score = np.zeros_like(x)
    score += np.where(x < lower_bound, (lower_bound - x) * BAD_SIDE_COEF, 0)
    score += np.where(x > upper_bound, (x - upper_bound) * GOOD_SIDE_COEF, 0)
    return -1 * score


def neutral_score(x: np.ndarray, lower_bound: float, upper_bound: float) -> np.ndarray:
    score = np.zeros_like(x)
    score += np.where(x < lower_bound, (x - lower_bound) * BAD_SIDE_COEF, 0)
    score += np.where(x > upper_bound, (upper_bound - x) * BAD_SIDE_COEF, 0)
    return -1 * score


class FactorwiseScoringStrategy(StreamlitReadyEnum):
    BY_FACTOR_TREND = "by factor trend"
    BY_OPTIMAL_RANGE = "by comparison wrt optimal range"
    BY_HISTORICAL_VALUES = "by comparison wrt historical values"
    BY_REFERENCE_VALUE = "by comparison wrt a reference point"
    RAW_VALUES = "raw values"

    def __call__(
        self,
        dataset: HazardDataset,
        dataset_historical: HazardDataset,
        optimal_ranges: Dict[str, List[float]],
        scaling: GeoPandasScalingStrategy,
    ) -> gpd.GeoDataFrame:
        if self is FactorwiseScoringStrategy.BY_FACTOR_TREND:
            scores = FactorwiseScoringStrategy._by_trend(dataset.df, dataset.factors)
        elif self is FactorwiseScoringStrategy.BY_HISTORICAL_VALUES:
            scores = FactorwiseScoringStrategy._by_historical_value(dataset_historical, dataset)
        elif self is FactorwiseScoringStrategy.BY_OPTIMAL_RANGE:
            scores = FactorwiseScoringStrategy._by_optimal_range_discrepancy(dataset.df, dataset.factors, optimal_ranges)
        elif self is FactorwiseScoringStrategy.BY_REFERENCE_VALUE:
            raise NotImplementedError("single point ref scoring not implemented yet")
        elif self is FactorwiseScoringStrategy.RAW_VALUES:
            scores = dataset.df.copy()
        else:
            raise ValueError()

        scores = scaling(scores)

        return scores

    def get_score_as_hazard_ds(
        self,
        dataset: HazardDataset,
        dataset_historical: HazardDataset,
        optimal_ranges: Dict[str, List[float]],
        scaling: GeoPandasScalingStrategy,
    ) -> HazardDataset:
        scores = self(
            dataset=dataset,
            dataset_historical=dataset_historical,
            optimal_ranges=optimal_ranges,
            scaling=scaling,
        )
        factors = [
            Factor(col, f"marginal score of {col}", trend=FactorTrend.HIGHER_BETTER, type=FactorType.CONTINUOUS, unit="none")
            for col in scores.columns
            if col != "geometry"
        ]
        return HazardDataset(df=scores, factors=factors)

    @staticmethod
    def _by_trend(df: gpd.GeoDataFrame, factors: List[Factor]) -> gpd.GeoDataFrame:
        """Each factor's score is its own value (or the inverse if higher is better)"""
        scores = df[["geometry"]].copy()
        for factor in factors:
            if factor.type == FactorType.BINARY:
                continue
            if factor.trend == FactorTrend.LOWER_BETTER:
                scores[factor.name] = -1.0 * df[factor.name]
            if factor.trend == FactorTrend.HIGHER_BETTER:
                scores[factor.name] = 1.0 * df[factor.name]
        return scores

    @staticmethod
    def _by_optimal_range_discrepancy(
        df: gpd.GeoDataFrame, factors: List[Factor], optimal_ranges: Dict[str, List[float]]
    ) -> gpd.GeoDataFrame:
        """Compare each factor to its optimal range and measure its discrepancy according to the factor's type"""
        scores = df[["geometry"]].copy()
        for factor in factors:
            if factor.name not in optimal_ranges:
                continue
            low_up_bounds = optimal_ranges[factor.name]
            if factor.trend == FactorTrend.LOWER_BETTER:
                scores[factor.name] = lower_better_score(df[factor.name], *low_up_bounds)
            if factor.trend == FactorTrend.HIGHER_BETTER:
                scores[factor.name] = higher_better_score(df[factor.name], *low_up_bounds)
            if factor.trend == FactorTrend.NEUTRAL:
                scores[factor.name] = neutral_score(df[factor.name], *low_up_bounds)

        return scores

    @staticmethod
    def _by_historical_value(ds_hist: HazardDataset, ds_proj: HazardDataset) -> pd.DataFrame:
        SUFFIX_REF = "L"
        SUFFIX_PROJ = "R"
        common_factors = sorted(list(set(ds_hist.factors).intersection(set(ds_proj.factors))))
        df_hist = ds_hist.df[[f.name for f in common_factors] + ["geometry"]]
        df_proj = ds_proj.df[[f.name for f in common_factors] + ["geometry"]]

        df = gpd.sjoin_nearest(df_hist, df_proj, lsuffix=SUFFIX_REF, rsuffix=SUFFIX_PROJ, how="inner", exclusive=True)

        scores = df[["geometry"]].copy()
        for factor in common_factors:
            if factor.is_binary():
                continue
            values_hist = df[factor.name + "_" + SUFFIX_REF]
            values_proj = df[factor.name + "_" + SUFFIX_PROJ]
            numerical_stability_value = values_hist[values_hist > 0].min()
            rel_diff = (values_proj - values_hist) / (numerical_stability_value + values_hist.abs())
            if factor.trend == FactorTrend.LOWER_BETTER:
                scores[factor.name] = -rel_diff
            elif factor.trend == FactorTrend.HIGHER_BETTER:
                scores[factor.name] = rel_diff
            elif factor.trend == FactorTrend.NEUTRAL:
                scores[factor.name] = rel_diff.abs()

        return scores
