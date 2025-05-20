from typing import Dict

import geopandas as gpd
import numpy as np
import pandas as pd

from heating_planner.back.data.base import HazardDataset
from heating_planner.back.data.model.factor import Factor
from heating_planner.back.streamlit_enums import StreamlitReadyEnum


class ScoringFusion(StreamlitReadyEnum):
    WEIGHTED_MEAN = "weighted mean of each factor's score"
    # RRF = "reciprocal rank fusion from each factor's rank"

    def __call__(self, scores: HazardDataset, coefs: Dict[Factor, float]) -> HazardDataset:
        gdf = scores.df[["geometry"]].copy()
        for factor in scores.factors:
            if factor.is_binary():
                gdf[factor.name] = scores.df[factor.name]
        score = self.call_on_df(scores.df, coefs)

        gdf["score"] = score
        return HazardDataset(df=gdf, factors=scores.factors)

    def call_on_df(self, df: pd.DataFrame, coefs: Dict[Factor, float]) -> np.ndarray:
        if self is ScoringFusion.WEIGHTED_MEAN:
            return ScoringFusion.weighted_mean(df, coefs)
        else:
            return ScoringFusion.reciprocal_rank_fusion(df, coefs)

    @staticmethod
    def weighted_mean(scores: pd.DataFrame, coefs: Dict[Factor, float]) -> np.ndarray:
        weighted_scores = np.zeros_like(scores.iloc[:, 0])
        for factor, coef in coefs.items():
            if factor.name not in scores.columns:
                continue
            weighted_scores += coef * scores[factor.name].values

        return weighted_scores

    @staticmethod
    def reciprocal_rank_fusion(scores: gpd.GeoDataFrame, coefs: Dict[Factor, float]) -> np.ndarray:
        weighted_scores = np.zeros_like(scores.iloc[:, 0])
        for factor, coef in coefs.items():
            if factor.name not in scores.columns:
                continue
            weighted_scores += -1 * coef / scores[factor.name].rank(method="min").values

        return weighted_scores
