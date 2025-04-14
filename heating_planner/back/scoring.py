import geopandas as gpd
from typing import Dict, List
from loguru import logger
import numpy as np
from heating_planner.back.tools import minmax_scale

from heating_planner.back.data.base import HazardDataset, LOWER_BETTER, HIGHER_BETTER, NEUTRAL

SCORE_COL = "score"

class HazardScoring:
    def __init__(self, hazard_dataset: HazardDataset):
        self.df = hazard_dataset.df
        self.columns_definition = hazard_dataset.columns_definition
        self.trend_preferences = hazard_dataset.trend_preferences
        self.ranked_df = self._compute_colwise_ranking()
    
    def _compute_colwise_ranking(self) -> gpd.GeoDataFrame:
        vars2ascending = {}
        for variable, trend in self.trend_preferences.items():
            if trend == HIGHER_BETTER:
                vars2ascending[variable] = False
            elif trend == LOWER_BETTER:
                vars2ascending[variable] = True
            else:
                logger.warning(f"Unknown trend preference for {variable}: {trend} -> skipping this variable")
            
        return self.df[vars2ascending.keys()].rank(ascending=vars2ascending.values())
    
    def compute_rrf_score(self, coefs: Dict[str, float] | None = None) -> gpd.GeoDataFrame:
        if coefs is None:
            coefs = {k: 1.0 for k in self.columns_definition.keys()}
        if missing_keys:=(set(coefs.keys()) - set(self.columns_definition.keys())):
            raise ValueError("Missing keys in coefs: " + str(missing_keys))
        
        coefs = {k: v for k, v in coefs.items() if self.trend_preferences[k] != NEUTRAL}
        
        rrf_score = -1 / np.dot(1 / self.ranked_df[coefs.keys()], np.array(list(coefs.values())))
        self.df[SCORE_COL] = minmax_scale(rrf_score)
        return self.df