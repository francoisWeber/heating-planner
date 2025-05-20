import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Tuple
from urllib.parse import urlparse

import geopandas as gpd
import pandas as pd
import requests
from loguru import logger
from shapely.geometry import Point

from heating_planner.back.data.model.factor import Factor
from heating_planner.back.geo.coder import geocoding
from heating_planner.back.geo.tools import make_geo_df
from heating_planner.crs import METRIC_CRS

SJOIN_MAX_DISTANCE_M = 8_000  # meters


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

    @staticmethod
    def resolve_path(path: str) -> str:
        """
        Resolves a local path or HTTP/HTTPS URL to a local file path.
        Downloads the file if necessary and returns the local path.

        Args:
            path: A local file path or URL

        Returns:
            A local file path that can be used to load the data

        Raises:
            FileNotFoundError: If the local file doesn't exist
            ValueError: If the URL scheme is not supported
            requests.RequestException: If there's an error downloading the URL
        """
        parsed = urlparse(path)
        scheme = parsed.scheme

        if scheme in ("http", "https"):
            # Download from URL
            logger.info(f"Downloading resource from {path}")
            with requests.get(path, stream=True, timeout=10) as r:
                r.raise_for_status()
                suffix = os.path.splitext(parsed.path)[-1]
                if not suffix:
                    # If no file extension in URL, try to determine from content type
                    content_type = r.headers.get("content-type", "")
                    if "text/plain" in content_type:
                        suffix = ".txt"
                    elif "application/json" in content_type:
                        suffix = ".json"
                    elif "application/zip" in content_type:
                        suffix = ".zip"
                    elif "application/octet-stream" in content_type:
                        # Try to determine format from content-disposition if available
                        content_disp = r.headers.get("content-disposition", "")
                        if "filename=" in content_disp:
                            filename = content_disp.split("filename=")[-1].strip("\"'")
                            suffix = os.path.splitext(filename)[-1]
                        else:
                            # Default to .dat for unknown binary content
                            suffix = ".dat"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                    logger.info(f"Downloaded resource to {f.name}")
                    return f.name

        elif scheme == "" or scheme == "file":
            # Local file
            local_path = parsed.path if scheme == "file" else path
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"File not found: {local_path}")
            return local_path

        else:
            raise ValueError(f"Unsupported path scheme: {scheme}")

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

    def explode_information(self) -> Tuple[pd.DataFrame, gpd.GeoDataFrame, List[Factor], Tuple[str, str, str]]:
        metadata = (self.path, self.model, self.scenario)
        df = self.df
        factors = self.factors
        ddf = df.copy()
        geometry = ddf.pop("geometry").to_frame("geometry")
        return ddf, geometry, factors, metadata

    def split_by_factor_type(self) -> Tuple["HazardDataset", "HazardDataset"]:
        df, geometry, factors, _ = self.explode_information()

        factors_binary = [factor for factor in factors if factor.is_binary()]
        factors_continuous = [factor for factor in factors if factor.is_continuous()]

        df_binary = df[[factor.name for factor in factors_binary]]
        df_continuous = df[[factor.name for factor in factors_continuous]]

        ds_binary = HazardDataset(df=make_geo_df(df_binary, geometry), factors=factors_binary)
        ds_continous = HazardDataset(df=make_geo_df(df_continuous, geometry), factors=factors_continuous)
        return ds_binary, ds_continous
