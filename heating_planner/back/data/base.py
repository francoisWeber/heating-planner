import json
import os
from dataclasses import dataclass
from enum import StrEnum
from typing import List

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from heating_planner.back.geo import geo_tool


class FactorTrend(StrEnum):
    HIGHER_BETTER = "higher_better"
    LOWER_BETTER = "lower_better"
    NEUTRAL = "neutral"

    @classmethod
    def from_string(cls, value: str) -> "FactorTrend":
        """Convert string to FactorType enum value"""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Invalid FactorType: {value}. Must be one of {[t.value for t in cls]}")


class FactorType(StrEnum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    BINARY = "binary"


@dataclass
class Factor:
    name: str
    description: str
    trend: FactorTrend
    type: FactorType

    def __hash__(self):
        return hash(self.name)


class DatasetFactors:
    def __init__(self, factors: List[Factor]):
        self.factors = factors
        self.factors_dict = {factor.name: factor for factor in factors}

    def __getitem__(self, key: str) -> Factor:
        return self.factors_dict[key]

    def __iter__(self):
        return iter(self.factors)

    def __len__(self):
        return len(self.factors)

    def __add__(self, other: "DatasetFactors"):
        if not isinstance(other, DatasetFactors):
            raise TypeError("Can only add two DatasetFactors objects")
        return DatasetFactors(self.factors + other.factors)


@dataclass
class HazardDataset:
    path: str | None = None
    df: gpd.GeoDataFrame | None = None
    model: str | None = None
    scenario: str | None = None
    factors: DatasetFactors | None = None

    @classmethod
    def load_from_path(cls, path: str):
        """Load dataset from path"""
        raise NotImplementedError("Must be implemented in subclass")

    def get_values(self) -> pd.DataFrame:
        return self.df[[factor.name for factor in self.factors]]

    def get_index_of_city(self, city: str) -> int:
        loc = geo_tool.geocode(city)
        point = Point(loc.longitude, loc.latitude)
        return self.df["geometry"].distance(point).idxmin()

    def __add__(self, other: "HazardDataset"):
        if not isinstance(other, HazardDataset):
            raise TypeError("Can only add two HazardDataset objects")
        if self.df is None or other.df is None:
            raise ValueError("Cannot add datasets with None df")

        merged_df = gpd.sjoin_nearest(self.df, other.df, how="inner").drop(columns=["index_right"])
        path = f"{self.path}_{other.path}"
        model = f"{self.model}_{other.model}"
        scenario = f"{self.scenario}_{other.scenario}"
        factors = self.factors + other.factors
        return HazardDataset(df=merged_df, path=path, model=model, scenario=scenario, factors=factors)

    def __radd__(self, other: "HazardDataset"):
        if other == None:
            return self
        else:
            return self.__add__(other)
