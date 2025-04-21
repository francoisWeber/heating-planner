from typing import Dict
import numpy as np
import geopandas as gpd
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from heating_planner.back.data.base import HazardDataset, Factor, FactorTrend
from heating_planner.back.streamlit_enums import StreamlitReadyEnum


class Contrast(StreamlitReadyEnum):
    DECREASE = "decrease"
    LEAVE = "leave"
    INCREASE = "increase"

    def __call__(self, values: np.ndarray) -> np.ndarray:
        contrast_factor = 1
        if self is Contrast.DECREASE:
            contrast_factor = 0.5
        if self is Contrast.INCREASE:
            contrast_factor = 2

        return np.power(values, contrast_factor)


class ScoreScalingStrategy(StreamlitReadyEnum):
    MINMAX = "min max"
    RANK = "rank"

    def __call__(self, values: np.ndarray) -> np.ndarray:
        if self is ScoreScalingStrategy.MINMAX:
            score = MinMaxScaler().fit_transform(values.reshape(-1, 1))
        else:
            score = pd.Series(values, name="score").rank(method="first", ascending=True, pct=True).values
        return score


def process_score(
    dataset: HazardDataset,
    score: np.ndarray,
    contrast: Contrast,
    scaling_strategy: ScoreScalingStrategy,
    binary_factor_infos: Dict[Factor, bool],
) -> gpd.GeoDataFrame:
    score = scaling_strategy(score)
    score = contrast(score)
    geo_score = dataset.df[["geometry"]].copy()
    geo_score["score"] = score.squeeze()

    for binary_factor, active in binary_factor_infos.items():
        if active:
            penalty = dataset.df[binary_factor.name].astype(float)
            if binary_factor.trend == FactorTrend.LOWER_BETTER:
                penalty = 1 - penalty
            geo_score["score"] *= penalty

    return geo_score
