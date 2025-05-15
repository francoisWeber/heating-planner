from typing import Dict
import numpy as np
import geopandas as gpd

from heating_planner.back.data.base import HazardDataset
from heating_planner.back.data.model.factor import Factor, FactorTrend
from heating_planner.back.streamlit_enums import StreamlitReadyEnum
from heating_planner.back.scoring.scaler import GeoPandasScalingStrategy


class Contrast(StreamlitReadyEnum):
    DECREASE = "decrease"
    LEAVE = "leave"
    INCREASE = "increase"

    def __call__(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        contrast_factor = 1
        if self is Contrast.DECREASE:
            contrast_factor = 0.5
        if self is Contrast.INCREASE:
            contrast_factor = 2

        df["score"] = np.power(df["score"].values, contrast_factor)

        return df


def process_score(
    dataset: HazardDataset,
    score: gpd.GeoDataFrame,
    contrast: Contrast,
    scaling_strategy: GeoPandasScalingStrategy,
    binary_factor_infos: Dict[Factor, bool],
) -> gpd.GeoDataFrame:
    score = scaling_strategy(score)
    score = contrast(score)

    for factor, is_active in binary_factor_infos.items():
        if is_active:
            penalty = dataset.df[["geometry", factor.name]]
            score = gpd.overlay(score, penalty)
            min_score = score.score.min()
            if factor.trend == FactorTrend.LOWER_BETTER:
                score.loc[score[factor.name], "score"] = min_score
            else:
                score.loc[np.logical_not(score[factor.name]), "score"] = min_score

            score.drop(columns=[factor.name])

    return score
