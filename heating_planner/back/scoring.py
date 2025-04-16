import geopandas as gpd
from typing import Dict, List
from loguru import logger
import numpy as np
import pandas as pd
from heating_planner.back.tools import minmax_scale

from heating_planner.back.data.base import HazardDataset, FactorType 

SCORE_COL = "score"

class ScoresFusion:
    @staticmethod
    def reciprocal_rank_fusion(df: pd.DataFrame, coefs: Dict[str, float], trends_preferences: Dict[str, FactorType]) -> np.ndarray:
        """
        Computes the Reciprocal Rank Fusion (RRF) score for a given DataFrame, variable-wise trends and coefficients.
        """
        usable_cols = []
        usable_coefs = []
        usable_ascending = []
        for col in df.columns:
            if coefs[col] == 0 or trends_preferences[col] not in [FactorType.HIGHER_BETTER, FactorType.LOWER_BETTER]:
                continue
            usable_cols.append(col)
            usable_coefs.append(coefs[col])
            usable_ascending.append(trends_preferences[col] == FactorType.LOWER_BETTER)
            
        rrf = 1 / (1 / pd.concat([df[col].rank(ascending=ascending) for col, ascending in zip(usable_cols, usable_ascending)], axis=1)).sum(axis=1)
        return rrf
    

class HazardScoring:
    def __init__(self, hazard_dataset: HazardDataset):
        self.df = hazard_dataset.df
        self.columns_definition = hazard_dataset.factors_definitions
        self.trend_preferences = hazard_dataset.factors_types
    
    def compute_rrf_score(self, coefs: Dict[str, float] | None = None) -> gpd.GeoDataFrame:
        if coefs is None:
            coefs = {k: 1.0 for k in self.columns_definition.keys()}
        if missing_keys:=(set(coefs.keys()) - set(self.columns_definition.keys())):
            raise ValueError("Missing keys in coefs: " + str(missing_keys))
        
        rrf = ScoresFusion.reciprocal_rank_fusion(self.df[coefs.keys()], coefs, self.trend_preferences)
        self.df[SCORE_COL] = minmax_scale(-1 * rrf**2)
        return self.df
    
        
        