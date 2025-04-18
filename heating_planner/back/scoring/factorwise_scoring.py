from typing import Dict, List
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from heating_planner.back.data.base import DatasetFactors, HazardDataset, FactorType

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


class FactorwiseScoring:
    @staticmethod
    def scale(df: pd.DataFrame, scaled: bool) -> pd.DataFrame:
        if scaled:
            scaler = MinMaxScaler()
            df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
        return df
    
    @staticmethod
    def by_factor_type(dataset: HazardDataset, scaled: bool = True) -> gpd.GeoDataFrame:
        scores = FactorwiseScoring._by_factor_type(dataset.df, dataset.factors)
        scores = FactorwiseScoring.scale(scores, scaled)
        scores = gpd.GeoDataFrame(pd.concat([dataset.df[["geometry"]], scores], axis=1))
        return scores
        
    @staticmethod
    def _by_factor_type(df: pd.DataFrame, factors: DatasetFactors) -> pd.DataFrame:
        """Each factor's score is its own value (or the inverse if higher is better)"""
        scores = pd.DataFrame()
        for factor in factors:
            if factor.type == FactorType.LOWER_BETTER:
                scores[factor.name] = -1.0 * df[factor.name]
            if factor.type == FactorType.HIGHER_BETTER:
                scores[factor.name] = 1.0 * df[factor.name]
        return scores


    @staticmethod
    def by_factor_optimal_range_discrepancy(dataset: HazardDataset, optimal_ranges: Dict[str, List[float]], scaled: bool = True) -> gpd.GeoDataFrame:
        scores = FactorwiseScoring._by_factor_optimal_range_discrepancy(dataset.df, dataset.factors, optimal_ranges)  
        scores = FactorwiseScoring.scale(scores, scaled)
        scores = gpd.GeoDataFrame(pd.concat([dataset.df[["geometry"]], scores], axis=1))
        return scores

    @staticmethod
    def _by_factor_optimal_range_discrepancy(
        df: pd.DataFrame, factors: DatasetFactors, optimal_ranges: Dict[str, List[float]]
    ) -> pd.DataFrame:
        """Compare each factor to its optimal range and measure its discrepancy according to the factor's type"""
        scores = pd.DataFrame()
        for factor in factors:
            if factor.name not in optimal_ranges:
                continue
            low_up_bounds = optimal_ranges[factor.name]
            if factor.type == FactorType.LOWER_BETTER:
                scores[factor.name] = lower_better_score(df[factor.name], *low_up_bounds)
            if factor.type == FactorType.HIGHER_BETTER:
                scores[factor.name] = higher_better_score(df[factor.name], *low_up_bounds)
            if factor.type == FactorType.NEUTRAL:
                scores[factor.name] = neutral_score(df[factor.name], *low_up_bounds)
        return scores


    @staticmethod
    def by_factor_reference_values(ref_dataset: HazardDataset, proj_dataset: HazardDataset, scaled: bool = True) -> gpd.GeoDataFrame:
        scores = FactorwiseScoring._by_factor_reference_values(ref_dataset.df, proj_dataset.df, proj_dataset.factors)
        scores = FactorwiseScoring.scale(scores, scaled)
        scores = gpd.GeoDataFrame(pd.concat([ref_dataset.df[["geometry"]], scores], axis=1))
        return scores

    @staticmethod
    def _by_factor_reference_values(df_ref: gpd.GeoDataFrame, df_proj: gpd.GeoDataFrame, factors: DatasetFactors) -> pd.DataFrame:
        SUFFIX_REF = "L"
        SUFFIX_PROJ = "R"
        common_factors = sorted(list(set(df_ref.columns).intersection(set(df_proj.columns))))
        df = gpd.sjoin_nearest(df_ref[common_factors], df_proj[common_factors], lsuffix=SUFFIX_REF, rsuffix=SUFFIX_PROJ)

        scores = pd.DataFrame()
        for factor in common_factors:
            if factor == "geometry":
                continue
            factor_ref = factor + "_" + SUFFIX_REF
            factor_proj = factor + "_" + SUFFIX_PROJ
            s = (df[factor_proj] - df[factor_ref]) / np.abs(df[factor_ref])
            if factors[factor].type == FactorType.LOWER_BETTER:
                scores[factor] = -1.0 * s
            elif factors[factor].type == FactorType.HIGHER_BETTER:
                scores[factor] = s
            elif factors[factor].type == FactorType.NEUTRAL:
                scores[factor] = np.abs(s)
            
        return scores


