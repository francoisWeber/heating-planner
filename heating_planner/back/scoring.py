import geopandas as gpd
from typing import Dict, List
import numpy as np
from heating_planner.back.tools import minmax_scale

from heating_planner.back.data.base import HazardDataset

SCORE_COL = "score"

class HazardScoring:
    def __init__(self, hazard_dataset: HazardDataset):
        self.df = hazard_dataset.df
        self.columns_definition = hazard_dataset.columns_definition
        self.ranked_df = self._compute_colwise_ranking()
    
    def _compute_colwise_ranking(self) -> gpd.GeoDataFrame:
        return self.df[self.columns_definition.keys()].rank()
    
    def compute_rrf_score(self, coefs: Dict[str, float] | None = None) -> gpd.GeoDataFrame:
        if coefs is None:
            coefs = {k: 1.0 for k in self.columns_definition.keys()}
        if missing_keys:=(set(coefs.keys()) - set(self.columns_definition.keys())):
            raise ValueError("Missing keys in coefs: " + str(missing_keys))
        rrf_score = -1 / np.dot(1 / self.ranked_df[coefs.keys()], np.array(list(coefs.values())))
        self.df[SCORE_COL] = minmax_scale(rrf_score)
        return self.df