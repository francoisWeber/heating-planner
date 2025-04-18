from typing import Dict
import geopandas as gpd
import numpy as np


class ScoringFusion:
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
