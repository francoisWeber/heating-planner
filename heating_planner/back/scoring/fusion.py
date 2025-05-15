from typing import Dict

import geopandas as gpd
import numpy as np
from heating_planner.back.streamlit_enums import StreamlitReadyEnum
from heating_planner.back.data.model.factor import Factor


class ScoringFusion(StreamlitReadyEnum):
    WEIGHTED_MEAN = "weighted mean of each factor's score"
    RRF = "reciprocal rank fusion from each factor's rank"

    def __call__(self, scores: gpd.GeoDataFrame, coefs: Dict[Factor, float]) -> np.ndarray:
        gdf = scores[["geometry"]].copy()
        if self is ScoringFusion.WEIGHTED_MEAN:
            score = ScoringFusion.weighted_mean(scores, coefs)
        else:
            score = ScoringFusion.reciprocal_rank_fusion(scores, coefs)

        gdf["score"] = score
        return gdf

    @staticmethod
    def weighted_mean(scores: gpd.GeoDataFrame, coefs: Dict[Factor, float]) -> np.ndarray:
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
