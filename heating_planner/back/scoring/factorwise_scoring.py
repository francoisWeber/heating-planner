from enum import StrEnum
from typing import Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from heating_planner.back.data.base import Factor, FactorTrend, HazardDataset

SCORE_COL = "score"
GOOD_SIDE_COEF = 0.1
BAD_SIDE_COEF = 1


def lower_better_score(x: np.ndarray, lower_bound: float, upper_bound: float) -> np.ndarray:
    score = np.zeros_like(x, dtype=float)
    score += np.where(x < lower_bound, (lower_bound - x) * GOOD_SIDE_COEF, 0)
    score += np.where(x > upper_bound, (x - upper_bound) * BAD_SIDE_COEF, 0)
    return score


def higher_better_score(x: np.ndarray, lower_bound: float, upper_bound: float) -> np.ndarray:
    score = np.zeros_like(x)
    score += np.where(x < lower_bound, (lower_bound - x) * BAD_SIDE_COEF, 0)
    score += np.where(x > upper_bound, (x - upper_bound) * GOOD_SIDE_COEF, 0)
    return score


def neutral_score(x: np.ndarray, lower_bound: float, upper_bound: float) -> np.ndarray:
    score = np.zeros_like(x)
    score += np.where(x < lower_bound, (lower_bound - x) * BAD_SIDE_COEF, 0)
    score += np.where(x > upper_bound, (x - upper_bound) * BAD_SIDE_COEF, 0)
    return score


class FactorwiseScoring(StrEnum):
    BY_FACTOR_TREND = "by factor trend"
    BY_OPTIMAL_RANGE = "by comparison wrt optimal range"
    BY_HISTORICAL_VALUES = "by comparison wrt historical values"
    BY_REFERENCE_VALUE = "by comparison wrt a reference"

    def __call__(
        self, dataset: HazardDataset, dataset_historical: HazardDataset, optimal_ranges: Dict[str, List[float]], scaled: bool = True
    ) -> gpd.GeoDataFrame:
        if self is FactorwiseScoring.BY_FACTOR_TREND:
            return FactorwiseScoring.by_trend(dataset, scaled)
        if self is FactorwiseScoring.BY_HISTORICAL_VALUES:
            return FactorwiseScoring.by_historical_value(dataset_historical, dataset, scaled)
        if self is FactorwiseScoring.BY_OPTIMAL_RANGE:
            return FactorwiseScoring.by_optimal_range_discrepancy(dataset, optimal_ranges, scaled)
        if self is FactorwiseScoring.BY_REFERENCE_VALUE:
            raise NotImplementedError("single point ref scoring not implemented yet")

    @classmethod
    def get_available_scorings(cls) -> List[str]:
        return [scoring.value for scoring in cls]

    @staticmethod
    def scale(df: pd.DataFrame, scaled: bool) -> pd.DataFrame:
        if scaled:
            scaler = MinMaxScaler()
            df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
        return df

    @staticmethod
    def by_trend(dataset: HazardDataset, scaled: bool = True) -> gpd.GeoDataFrame:
        scores = FactorwiseScoring._by_trend(dataset.df, dataset.factors)
        scores = FactorwiseScoring.scale(scores, scaled)
        scores = gpd.GeoDataFrame(pd.concat([dataset.df[["geometry"]], scores], axis=1))
        return scores

    @staticmethod
    def _by_trend(df: pd.DataFrame, factors: List[Factor]) -> pd.DataFrame:
        """Each factor's score is its own value (or the inverse if higher is better)"""
        scores = pd.DataFrame()
        for factor in factors:
            if factor.trend == FactorTrend.LOWER_BETTER:
                scores[factor.name] = -1.0 * df[factor.name]
            if factor.trend == FactorTrend.HIGHER_BETTER:
                scores[factor.name] = 1.0 * df[factor.name]
        return scores

    @staticmethod
    def by_optimal_range_discrepancy(
        dataset: HazardDataset, optimal_ranges: Dict[str, List[float]], scaled: bool = True
    ) -> gpd.GeoDataFrame:
        scores = FactorwiseScoring._by_optimal_range_discrepancy(dataset.df, dataset.factors, optimal_ranges)
        scores = FactorwiseScoring.scale(scores, scaled)
        scores = gpd.GeoDataFrame(pd.concat([dataset.df[["geometry"]], scores], axis=1))
        return scores

    @staticmethod
    def _by_optimal_range_discrepancy(df: pd.DataFrame, factors: List[Factor], optimal_ranges: Dict[str, List[float]]) -> pd.DataFrame:
        """Compare each factor to its optimal range and measure its discrepancy according to the factor's type"""
        scores = pd.DataFrame()
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
    def by_historical_value(dataset_hist: HazardDataset, dataset_proj: HazardDataset, scaled: bool = True) -> gpd.GeoDataFrame:
        scores = FactorwiseScoring._by_historical_value(dataset_hist.df, dataset_proj.df, dataset_proj.factors)
        scores = FactorwiseScoring.scale(scores, scaled)
        scores = gpd.GeoDataFrame(pd.concat([dataset_hist.df[["geometry"]], scores], axis=1))
        return scores

    @staticmethod
    def _by_historical_value(df_ref: gpd.GeoDataFrame, df_proj: gpd.GeoDataFrame, factors: List[Factor]) -> pd.DataFrame:
        SUFFIX_REF = "L"
        SUFFIX_PROJ = "R"
        common_factors = sorted(list(set(df_ref.columns).intersection(set(df_proj.columns))))
        df = gpd.sjoin_nearest(df_ref[common_factors], df_proj[common_factors], lsuffix=SUFFIX_REF, rsuffix=SUFFIX_PROJ)
        name2factor = {factor.name: factor for factor in factors}

        scores = pd.DataFrame()
        for factor_name in common_factors:
            if factor_name == "geometry":
                continue
            factor_ref = factor_name + "_" + SUFFIX_REF
            factor_proj = factor_name + "_" + SUFFIX_PROJ
            s = (df[factor_proj] - df[factor_ref]) / np.abs(df[factor_ref])
            if name2factor[factor_name].trend == FactorTrend.LOWER_BETTER:
                scores[factor_name] = -1.0 * s
            elif name2factor[factor_name].trend == FactorTrend.HIGHER_BETTER:
                scores[factor_name] = s
            elif name2factor[factor_name].trend == FactorTrend.NEUTRAL:
                scores[factor_name] = np.abs(s)

        return scores
