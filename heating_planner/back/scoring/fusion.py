from enum import StrEnum
from typing import Dict, List

import geopandas as gpd
import numpy as np


class ScoringFusion(StrEnum):
    WEIGHTED_MEAN = "weighted mean of each factor's score"
    RRF = "reciprocal rank fusion from each factor's rank"

    def __call__(self, scores: gpd.GeoDataFrame, coefs: Dict[str, float]) -> gpd.GeoDataFrame:
        if self is ScoringFusion.WEIGHTED_MEAN:
            return ScoringFusion.weighted_mean(scores, coefs)
        if self is ScoringFusion.RRF:
            return ScoringFusion.reciprocal_rank_fusion(scores, coefs)

    @classmethod
    def get_available_fusions(cls) -> List[str]:
        return [fusion.value for fusion in cls]

    @staticmethod
    def weighted_mean(scores: gpd.GeoDataFrame, coefs: Dict[str, float]) -> gpd.GeoDataFrame:
        weighted_scores = []
        for factor, coef in coefs.items():
            if factor not in scores.columns:
                continue
            weighted_scores.append(coef * scores[factor].values)
        geo_score = scores[["geometry"]].copy()
        geo_score["score"] = np.sum(weighted_scores, axis=1)

    @staticmethod
    def reciprocal_rank_fusion(scores: gpd.GeoDataFrame, coefs: Dict[str, float]) -> gpd.GeoDataFrame:
        weighted_scores = []
        for factor, coef in coefs.items():
            if factor not in scores.columns:
                continue
            weighted_scores.append(coef / scores[factor].rank(method="min").values)

        geo_score = scores[["geometry"]].copy()
        geo_score["score"] = np.sum(weighted_scores, axis=1)
