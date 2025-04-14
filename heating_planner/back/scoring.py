import geopandas as gpd
from typing import List
import numpy as np
from abc import abstractmethod


class DriasScoring:
    def __init__(self, datasets: List[gpd.GeoDataFrame]):
        self.dfs = dfs
        self.df = self._merge_gdf(dfs)
        self.columns_definition = None

    def _merge_gdf(self, dfs: List[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
        merged = dfs[0]
        for df in dfs[1:]:
            merged = gpd.sjoin_nearest(merged, df, how="inner").drop(columns=["index_right"])
        return merged

    def _merge_columns_definitions(self, dfs: List[gpd.GeoDataFrame]) -> dict:
        merged = {}
        for df in dfs:
            merged.update(df.columns_definition)
        return merged
