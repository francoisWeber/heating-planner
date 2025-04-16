from heating_planner.back.geo import geo_tool
import geopandas as gpd
from shapely.geometry import Point
import json
import os
from enum import StrEnum

class FactorType(StrEnum):
    HIGHER_BETTER = "higher_better"
    LOWER_BETTER = "lower_better"
    NEUTRAL = "neutral"
    BINARY = "binary"

TREND_PREFERENCES_FNAME = "trend_preference_per_var.json"

class HazardDataset:
    def __init__(self, path: str | None = None, df: gpd.GeoDataFrame | None = None, factors_definitions: dict | None = None, model: str | None = None, scenario: str | None = None, is_boolean: bool = False, factors_types: dict | None = None):
        self.path = path
        self.df = df
        self.factors_definitions = factors_definitions
        self.model = model
        self.scenario = scenario
        self.is_boolean = is_boolean
        self.factors_types = factors_types
    
    @classmethod
    def load_from_path(cls, path: str):
        """Load dataset from path"""
        raise NotImplementedError("Must be implemented in subclass")
    
    @staticmethod
    def find_and_load_factors_types(path):
        if os.path.isdir(path):
            json_path = os.path.join(path, TREND_PREFERENCES_FNAME)
        elif os.path.isfile(path):
            json_path = os.path.join(os.path.dirname(path), TREND_PREFERENCES_FNAME)
        else:
            raise FileNotFoundError(f"Path {path} does not exist")

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Trend preferences file not found at {json_path}")

        with open(json_path, 'r') as file:
            return json.load(file)
    

    def get_index_of_city(self, city: str) -> int:
        loc = geo_tool.geocode(city)
        point = Point(loc.longitude, loc.latitude)
        return self.df['geometry'].distance(point).idxmin()
    
    def __add__(self, other: "HazardDataset"):
        if not isinstance(other, HazardDataset):
            raise TypeError("Can only add two HazardDataset objects")
        if self.df is None or other.df is None:
            raise ValueError("Cannot add datasets with None df")
        
        merged_df = gpd.sjoin_nearest(self.df, other.df, how="inner").drop(columns=["index_right"])
        path = f"{self.path}_{other.path}"
        factors_definitions = {**self.factors_definitions, **other.factors_definitions}
        model = f"{self.model}_{other.model}"
        scenario = f"{self.scenario}_{other.scenario}"
        factors_types = {**self.factors_types, **other.factors_types}
        is_boolean = self.is_boolean and other.is_boolean
        return HazardDataset(df=merged_df, path=path, factors_definitions=factors_definitions, model=model, scenario=scenario, factors_types=factors_types, is_boolean=is_boolean)
    
    def __radd__(self, other: "HazardDataset"):
        if other == None:
            return self
        else:
            return self.__add__(other)