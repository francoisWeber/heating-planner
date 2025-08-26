from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
from loguru import logger

from heating_planner.back.data.base import HazardDataset
from heating_planner.back.data.model.factor import Factor, FactorTrend, FactorType
from heating_planner.back.geo.tools import make_geo_df
from heating_planner.back.streamlit_enums import StreamlitReadyEnum

SCORE_COL = "score"
GOOD_SIDE_COEF = 0.1
BAD_SIDE_COEF = 1


def lower_better_score(
    x: np.ndarray, lower_bound: float, upper_bound: float
) -> np.ndarray:
    score = np.zeros_like(x, dtype=float)
    score += np.where(x < lower_bound, (lower_bound - x) * GOOD_SIDE_COEF, 0)
    score += np.where(x > upper_bound, (x - upper_bound) * BAD_SIDE_COEF, 0)
    return -1 * score


def higher_better_score(
    x: np.ndarray, lower_bound: float, upper_bound: float
) -> np.ndarray:
    score = np.zeros_like(x)
    score += np.where(x < lower_bound, (lower_bound - x) * BAD_SIDE_COEF, 0)
    score += np.where(x > upper_bound, (x - upper_bound) * GOOD_SIDE_COEF, 0)
    return -1 * score


def neutral_score(x: np.ndarray, lower_bound: float, upper_bound: float) -> np.ndarray:
    score = np.zeros_like(x)
    score += np.where(x < lower_bound, (x - lower_bound) * BAD_SIDE_COEF, 0)
    score += np.where(x > upper_bound, (upper_bound - x) * BAD_SIDE_COEF, 0)
    return -1 * score


class FactorsScoringStrategy(StreamlitReadyEnum):
    BY_FACTOR_TREND = "by factor trend"
    BY_OPTIMAL_RANGE = "by comparison wrt optimal range"
    BY_HISTORICAL_VALUES = "by comparison wrt historical values"
    BY_REFERENCE_VALUE = "by comparison wrt a reference point"
    RAW_VALUES = "raw values"

    def __call__(
        self,
        dataset_proj: HazardDataset,
        dataset_hist: HazardDataset,
        optimal_ranges: Dict[str, List[float]],
    ) -> HazardDataset:
        if self is FactorsScoringStrategy.BY_FACTOR_TREND:
            scores, new_factors = FactorsScoringStrategy._by_trend(
                dataset_proj.df, dataset_proj.factors
            )
        elif self is FactorsScoringStrategy.BY_HISTORICAL_VALUES:
            scores, new_factors = FactorsScoringStrategy._by_historical_value(
                dataset_hist, dataset_proj
            )
        elif self is FactorsScoringStrategy.BY_OPTIMAL_RANGE:
            scores, new_factors = FactorsScoringStrategy._by_optimal_range_discrepancy(
                dataset_proj.df, dataset_proj.factors, optimal_ranges
            )
        elif self is FactorsScoringStrategy.BY_REFERENCE_VALUE:
            raise NotImplementedError("single point ref scoring not implemented yet")
        elif self is FactorsScoringStrategy.RAW_VALUES:
            return dataset_proj
        else:
            raise ValueError()

        return HazardDataset(df=scores, factors=new_factors)

    @staticmethod
    def _by_trend(
        df: gpd.GeoDataFrame, factors: List[Factor]
    ) -> Tuple[gpd.GeoDataFrame, List[Factor]]:
        """Each factor's score is its own value (or the inverse if higher is better)"""
        scores = df[["geometry"]].copy()
        new_factors = []
        for factor in factors:
            if factor.type == FactorType.BINARY:  # skip binary factors
                continue
            if factor.trend == FactorTrend.LOWER_BETTER:
                scores[factor.name] = -1.0 * df[factor.name]
            elif factor.trend == FactorTrend.HIGHER_BETTER:
                scores[factor.name] = 1.0 * df[factor.name]
            else:
                logger.error(f"Factor {factor.name} has an unhandled specs {factor}")
            score_factor = factor.copy()
            score_factor.trend = FactorTrend.HIGHER_BETTER
            new_factors.append(score_factor)
        return scores, new_factors

    @staticmethod
    def _by_optimal_range_discrepancy(
        df: gpd.GeoDataFrame,
        factors: List[Factor],
        optimal_ranges: Dict[str, List[float]],
    ) -> Tuple[gpd.GeoDataFrame, List[Factor]]:
        """Compare each factor to its optimal range and measure its discrepancy according to the factor's type"""
        scores = df[["geometry"]].copy()
        new_factors = []
        for factor in factors:
            if factor.name not in optimal_ranges or factor.is_binary():
                continue
            low_up_bounds = optimal_ranges[factor.name]
            if factor.trend == FactorTrend.LOWER_BETTER:
                scores[factor.name] = lower_better_score(
                    df[factor.name], *low_up_bounds
                )
            if factor.trend == FactorTrend.HIGHER_BETTER:
                scores[factor.name] = higher_better_score(
                    df[factor.name], *low_up_bounds
                )
            if factor.trend == FactorTrend.NEUTRAL:
                scores[factor.name] = neutral_score(df[factor.name], *low_up_bounds)
            score_factor = factor.copy()
            score_factor.trend = FactorTrend.HIGHER_BETTER
            new_factors.append(score_factor)

        return scores, new_factors

    @staticmethod
    def _by_historical_value(
        ds_hist: HazardDataset, ds_proj: HazardDataset
    ) -> Tuple[gpd.GeoDataFrame, List[Factor]]:
        common_factors = sorted(
            list(set(ds_hist.factors).intersection(set(ds_proj.factors)))
        )
        common_factors_continuous = [f for f in common_factors if f.is_continuous()]

        df_hist = ds_hist.df[[f.name for f in common_factors_continuous]]
        df_proj = ds_proj.df[[f.name for f in common_factors_continuous]]

        stability_per_factor = df_hist.apply(
            lambda x: np.nanmin(np.where(x.to_numpy() > 0, x.to_numpy(), np.nan)) / 10
        )

        relative_evolution = (df_proj - df_hist) / (df_hist + stability_per_factor)

        new_factors = []
        for factor in common_factors:
            if factor.is_binary():
                continue
            if factor.trend == FactorTrend.LOWER_BETTER:
                relative_evolution[factor.name] = -relative_evolution[factor.name]

            score_factor = factor.copy()
            score_factor.trend = FactorTrend.HIGHER_BETTER
            new_factors.append(score_factor)

        relative_evolution = make_geo_df(relative_evolution, ds_hist.df["geometry"])

        return relative_evolution, new_factors
