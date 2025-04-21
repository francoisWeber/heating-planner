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
            score = pd.Series(values.squeeze(), name="score").rank(method="first", ascending=True, pct=True).values
        return score


def process_score(
    dataset: HazardDataset,
    score: gpd.GeoDataFrame,
    contrast: Contrast,
    scaling_strategy: ScoreScalingStrategy,
    binary_factor_infos: Dict[Factor, bool],
) -> gpd.GeoDataFrame:
    geo_score = score.pop("geometry").to_frame()
    score = scaling_strategy(score.values)
    score = contrast(score)
    geo_score["score"] = score.squeeze()

    for factor, is_active in binary_factor_infos.items():
        if is_active:
            penalty = dataset.df[["geometry", factor.name]]
            geo_score = gpd.sjoin_nearest(geo_score, penalty, how="inner", exclusive=True, max_distance=10_000)
            if factor.trend == FactorTrend.HIGHER_BETTER:
                geo_score[geo_score[factor.name] is True, "score"] = 0.0
            else:
                geo_score[geo_score[factor.name] is False, "score"] = 0.0
                
            geo_score.drop(columns=[factor.name, "index_right"])
        
    return geo_score
