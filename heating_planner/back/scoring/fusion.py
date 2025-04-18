from enum import StrEnum
from typing import Dict, List

import geopandas as gpd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class ScoringFusion(StrEnum):
    WEIGHTED_MEAN = "weighted mean of each factor's score"
    RRF = "reciprocal rank fusion from each factor's rank"

    def __call__(self, scores: gpd.GeoDataFrame, coefs: Dict[str, float]) -> gpd.GeoDataFrame:
        if self is ScoringFusion.WEIGHTED_MEAN:
            score = ScoringFusion.weighted_mean(scores, coefs)
        elif self is ScoringFusion.RRF:
            score = ScoringFusion.reciprocal_rank_fusion(scores, coefs)
        else:
            raise ValueError()
        
        score = MinMaxScaler().fit_transform(score.reshape(-1, 1))
        geo_score = scores[["geometry"]].copy()
        geo_score["score"] = score.squeeze()
        return geo_score

    @classmethod
    def get_available_fusions(cls) -> List["ScoringFusion"]:
        return [fusion for fusion in cls]

    @staticmethod
    def weighted_mean(scores: gpd.GeoDataFrame, coefs: Dict[str, float]) -> gpd.GeoDataFrame:
        weighted_scores = np.zeros_like(scores.iloc[:, 0])
        for factor, coef in coefs.items():
            if factor not in scores.columns:
                continue
            weighted_scores += coef * scores[factor].values
            
        return weighted_scores

    @staticmethod
    def reciprocal_rank_fusion(scores: gpd.GeoDataFrame, coefs: Dict[str, float]) -> gpd.GeoDataFrame:
        weighted_scores = np.zeros_like(scores.iloc[:, 0])
        for factor, coef in coefs.items():
            if factor not in scores.columns:
                continue
            weighted_scores += coef / scores[factor].rank(method="min").values

        return weighted_scores
