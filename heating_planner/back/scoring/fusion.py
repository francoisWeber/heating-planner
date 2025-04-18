from enum import StrEnum
from typing import Dict, List

import pandas as pd
import numpy as np
from heating_planner.back.streamlit_enums import StreamlitReadyEnum


class ScoringFusion(StreamlitReadyEnum):
    WEIGHTED_MEAN = "weighted mean of each factor's score"
    RRF = "reciprocal rank fusion from each factor's rank"

    def __call__(self, scores: pd.DataFrame, coefs: Dict[str, float], increase_contrast: bool = False) -> np.ndarray:
        if self is ScoringFusion.WEIGHTED_MEAN:
            return ScoringFusion.weighted_mean(scores, coefs)
        if self is ScoringFusion.RRF:
            return ScoringFusion.reciprocal_rank_fusion(scores, coefs)

    @staticmethod
    def weighted_mean(scores: pd.DataFrame, coefs: Dict[str, float]) -> np.ndarray:
        weighted_scores = np.zeros_like(scores.iloc[:, 0])
        for factor, coef in coefs.items():
            if factor not in scores.columns:
                continue
            weighted_scores += coef * scores[factor].values

        return weighted_scores

    @staticmethod
    def reciprocal_rank_fusion(scores: pd.DataFrame, coefs: Dict[str, float]) -> np.ndarray:
        weighted_scores = np.zeros_like(scores.iloc[:, 0])
        for factor, coef in coefs.items():
            if factor not in scores.columns:
                continue
            weighted_scores += coef / scores[factor].rank(method="min").values

        return weighted_scores
