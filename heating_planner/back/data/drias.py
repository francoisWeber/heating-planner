from io import StringIO
from typing import Dict

import geopandas as gpd
import pandas as pd
from loguru import logger

from heating_planner.back.data.base import METRIC_CRS, HazardDataset
from heating_planner.back.data.model.factor import Factor, FactorTrend, FactorType

# Hardcoded trend preferences based on variable meanings
DRIAS_TREND_PREFERENCES = {
    # Temperature variables
    "nortmm_yr": FactorTrend.LOWER_BETTER,  # Annual average temperature
    "nortmm_seas_jja": FactorTrend.LOWER_BETTER,  # Summer average temperature
    "nortmm_seas_djf": FactorTrend.HIGHER_BETTER,  # Winter average temperature
    "nortxm_seas_jja": FactorTrend.LOWER_BETTER,  # Summer maximum temperature
    "nortx35d_yr": FactorTrend.LOWER_BETTER,  # Number of days with Tx >= 35°C
    "nortx30d_yr": FactorTrend.LOWER_BETTER,  # Number of days with Tx >= 30°C
    "nortr_yr": FactorTrend.LOWER_BETTER,  # Number of tropical nights
    # Precipitation variables
    "norrr_yr": FactorTrend.LOWER_BETTER,  # Annual precipitation
    "norrr_seas_jja": FactorTrend.HIGHER_BETTER,  # Summer precipitation
    "norrr_seas_djf": FactorTrend.HIGHER_BETTER,  # Winter precipitation
    "norrrq99_yr": FactorTrend.LOWER_BETTER,  # Remarkable daily precipitation (99th percentile)
    "norrx1d_yr": FactorTrend.LOWER_BETTER,  # Extreme precipitation intensity
    "norrrq99refd_yr": FactorTrend.LOWER_BETTER,  # Frequency of remarkable daily precipitation
    # Other indices
    "norifm40_yr": FactorTrend.LOWER_BETTER,  # Fire weather indicator (days > 40)
    "norswi04_yr": FactorTrend.LOWER_BETTER,  # Number of days with SWI < 0.4 (soil dryness)
}

DRIAS_EXPORT_SECTION_MODEL = 1
DRIAS_EXPORT_SECTION_SCENARIO = 1
DRIAS_EXPORT_SECTION_COLUMNS_DEF = 3
DRIAS_EXPORT_SECTION_DATA = 4


def normalize_colname(colname: str) -> str:
    return colname.replace("#", "").strip().lower()


class DriasDataset(HazardDataset):
    @classmethod
    def load_from_path(cls, path):
        local_path = cls.resolve_path(path)
        with open(local_path, "r") as f:
            raw_lines = f.readlines()

        sections_loc = cls.detect_sections_loc(raw_lines)

        df = cls.get_df_from_lines(raw_lines, sections_loc)
        model = cls.get_model_from_lines(raw_lines, sections_loc)
        scenario = cls.get_scenario_from_lines(raw_lines, sections_loc)

        factors_trends = DRIAS_TREND_PREFERENCES
        factors_descriptions = cls.get_factors_description_from_lines(
            raw_lines, sections_loc
        )
        factors = []
        for name in df.columns:
            trend = factors_trends.get(name)
            if trend is None:
                continue
            description = factors_descriptions.get(name)
            if description is None:
                continue
            descr_parts = description.split("(")
            description = descr_parts[0].strip()
            unit = "".join(descr_parts[1:])[:-1] if len(descr_parts) > 1 else "no unit"

            if trend is None or description is None:
                continue
            # No need to convert string to enum since we're already using FactorTrend enum values
            factors.append(
                Factor(
                    name=name,
                    description=description,
                    trend=trend,
                    type=FactorType.CONTINUOUS,
                    unit=unit,
                )
            )

        return cls(path=path, df=df, model=model, scenario=scenario, factors=factors)

    def __hash__(self):
        return hash(self.path)

    @staticmethod
    def get_factors_description_from_lines(
        raw_lines: list[str], sections_loc
    ) -> Dict[str, str]:
        sec_id = DRIAS_EXPORT_SECTION_COLUMNS_DEF
        lines = raw_lines[sections_loc[sec_id] + 2 : sections_loc[sec_id + 1]]
        columns = {}
        for line in lines:
            (k, v) = line.replace("#", "").strip().split(" : ")
            k = normalize_colname(k)
            columns[k] = v
        if len(columns) == 0:
            logger.warning("No columns found in DRIAS export file")
            return None
        return columns

    @staticmethod
    def detect_sections_loc(raw_lines: list[str]) -> dict[int, int]:
        return [i for i, line in enumerate(raw_lines) if line.startswith("#---")]

    @staticmethod
    def get_model_from_lines(raw_lines: list[str], sections_loc) -> pd.DataFrame:
        sec_id = DRIAS_EXPORT_SECTION_MODEL
        lines = raw_lines[sections_loc[sec_id] + 1 : sections_loc[sec_id + 1]]
        for line in lines:
            if line.startswith("# Modele"):
                return line.split(":")[1].strip()
        logger.warning("Model not found in DRIAS export file")

    @staticmethod
    def get_scenario_from_lines(raw_lines: list[str], sections_loc) -> pd.DataFrame:
        sec_id = DRIAS_EXPORT_SECTION_SCENARIO
        lines = raw_lines[sections_loc[sec_id] + 1 : sections_loc[sec_id + 1]]
        for i, line in enumerate(lines):
            if line.startswith("# Scenario"):
                line = lines[i + 1]
                return line.replace("#", "").strip()
        logger.warning("Scenario not found in DRIAS export file")

    @staticmethod
    def get_df_from_lines(raw_lines: list[str], sections_loc) -> pd.DataFrame:
        lines = raw_lines[sections_loc[DRIAS_EXPORT_SECTION_DATA] + 2 :]
        data = StringIO("".join(lines))
        df = (
            pd.read_csv(data, sep=";")
            .dropna(axis=0, subset="Contexte")
            .dropna(axis=1, how="all")
            .rename(columns=normalize_colname)
        )
        df = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326"
        )
        df = df.to_crs(METRIC_CRS)
        df = df.drop(columns=["longitude", "latitude", "point", "contexte"])
        return df
