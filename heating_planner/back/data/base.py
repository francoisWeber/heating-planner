from dataclasses import dataclass
from enum import StrEnum
from typing import Dict, List
from loguru import logger
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from heating_planner.back.geo.coder import geocoding
from heating_planner.crs import METRIC_CRS

SJOIN_MAX_DISTANCE_M = 8_000  # meters


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
    unit: str

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"{self.name} ({self.type}): {self.description[:50]}... "

    def is_binary(self):
        return self.type == FactorType.BINARY

    def is_continuous(self):
        return self.type == FactorType.CONTINUOUS

    def __eq__(self, other):
        return self.name == other.name

    def __lt__(self, other):
        return self.name < other.name

    def __le__(self, other):
        return self.name <= other.name


@dataclass
class HazardDataset:
    path: str | None = None
    df: gpd.GeoDataFrame | None = None
    model: str | None = None
    scenario: str | None = None
    factors: List[Factor] | None = None

    _name2factors: Dict[str, Factor] | None = None

    def __post_init__(self):
        self._name2factors = {f.name: f for f in self.factors} if self.factors else {}
        if not (crs := self.df.geometry.crs).is_projected:
            logger.warning(f"Using a non-projected CRS {crs}. Prefer {METRIC_CRS}")

    def get_factor(self, name: str) -> Factor:
        return self._name2factors.get(name)

    @classmethod
    def load_from_path(cls, path: str):
        """Load dataset from path"""
        raise NotImplementedError("Must be implemented in subclass")

    def get_values(self) -> pd.DataFrame:
        return self.df[[factor.name for factor in self.factors]]

    def get_index_of_city(self, city: str) -> int:
        loc = geocoding.geocode(city)
        point = Point(loc.longitude, loc.latitude)
        return self.df["geometry"].to_crs(epsg=4326).distance(point).idxmin()  # re-map to GPS CRS for comparison

    def __add__(self, other: "HazardDataset"):
        if not isinstance(other, HazardDataset):
            raise TypeError("Can only add two HazardDataset objects")
        if self.df is None or other.df is None:
            raise ValueError("Cannot add datasets with None df")

        merged_df = gpd.sjoin_nearest(self.df, other.df, how="inner", max_distance=SJOIN_MAX_DISTANCE_M).drop(columns=["index_right"])
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


@dataclass
class MappableFactors:
    continuous: List[Factor] | None = None
    discretes: List[Factor] | None = None
    binaries: List[Factor] | None = None

    @classmethod
    def from_hazard_datasets(cls, dataset1: HazardDataset, dataset2: HazardDataset):
        common_factors = list(set(dataset1.factors).intersection(set(dataset2.factors)))
        binaries: List[Factor] = []
        continuous: List[Factor] = []
        discretes: List[Factor] = []
        for factor in common_factors:
            if factor.type == FactorType.BINARY:
                binaries.append(factor)
            elif factor.type == FactorType.CONTINUOUS:
                continuous.append(factor)
            elif factor.type == FactorType.DISCRETE:
                discretes.append(factor)
            else:
                raise ValueError(f"Unknown type for {factor=}")

        return cls(continuous=continuous, discretes=discretes, binaries=binaries)

    @property
    def weightables(self):
        return self.continuous + self.discretes
