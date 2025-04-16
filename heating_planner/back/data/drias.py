from typing import List
from matplotlib import pyplot as plt
import pandas as pd
from loguru import logger
from io import StringIO
from heating_planner.back.data.base import HazardDataset


import geopandas as gpd



DRIAS_EXPORT_SECTION_MODEL = 1
DRIAS_EXPORT_SECTION_SCENARIO = 1
DRIAS_EXPORT_SECTION_COLUMNS_DEF = 3
DRIAS_EXPORT_SECTION_DATA = 4


def normalize_colname(colname: str) -> str:
    return colname.replace("#", "").strip().lower()


class DriasDataset(HazardDataset):

    @classmethod
    def load_from_path(cls, path):
        with open(path, "r") as f:
            raw_lines = f.readlines()

        sections_loc = cls.detect_sections_loc(raw_lines)


        model = cls.get_model_from_lines(raw_lines, sections_loc)
        scenario = cls.get_scenario_from_lines(raw_lines, sections_loc)
        factors_definitions = cls.get_factors_definition_from_lines(raw_lines, sections_loc)
        df = cls.get_df_from_lines(raw_lines, sections_loc)
        factors_types = HazardDataset.find_and_load_factors_types(path)
        
        return cls(
            path=path,
            df=df,
            factors_definitions=factors_definitions,
            model=model,
            scenario=scenario,
            factors_types=factors_types
        )

    def __hash__(self):
        return hash(self.path)

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
    def get_factors_definition_from_lines(raw_lines: list[str], sections_loc) -> pd.DataFrame:
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
    def get_df_from_lines(raw_lines: list[str], sections_loc) -> pd.DataFrame:
        lines = raw_lines[sections_loc[DRIAS_EXPORT_SECTION_DATA] + 2 :]
        data = StringIO("".join(lines))
        df = pd.read_csv(data, sep=";").dropna(axis=0, subset="Contexte").dropna(axis=1, how="all").rename(columns=normalize_colname)
        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
        return df