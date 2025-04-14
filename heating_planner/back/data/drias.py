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
        columns_definition = cls.get_columns_definition_from_lines(raw_lines, sections_loc)
        df = cls.get_df_from_lines(raw_lines, sections_loc)
        
        return cls(
            path=path,
            df=df,
            columns_definition=columns_definition,
            model=model,
            scenario=scenario,
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
    def get_columns_definition_from_lines(raw_lines: list[str], sections_loc) -> pd.DataFrame:
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

    def display_histograms(self, ref_locations: List[int | str] | None = None, **hist_kwargs):
        ref_locations = ref_locations or []
        ref_loc_names = [loc if isinstance(loc, str) else f"ref-{i}" for i, loc in enumerate(ref_locations)]
        ref_loc_lines = [loc if isinstance(loc, int) else self.get_index_of_city(loc) for loc in ref_locations]
        ddf = self.df.drop(columns=["point", "latitude", "longitude", "contexte"])
        num_columns = len(ddf.columns)
        num_rows = 1 + (num_columns + 3) // 4  # Calculate the number of rows needed for 3 columns
        fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
        axes = axes.flatten()

        _hist_kwargs = {"bins": 30, "alpha": 0.7, "density": True, "log": True}
        _hist_kwargs.update(hist_kwargs)

        for i, column in enumerate(ddf.columns):
            ddf[column].hist(ax=axes[i], **_hist_kwargs)
            for ref_line, ref_city in zip(ref_loc_lines, ref_loc_names):
                axes[i].vlines(
                    self.df.iloc[ref_line][column],
                    ymin=0,
                    ymax=axes[i].get_ylim()[1],
                    label=ref_city,
                    colors=plt.cm.tab10(ref_loc_lines.index(ref_line) % 10),  # Use a colormap for distinct colors
                )
            axes[i].set_title(self.columns_definition[column][:40])
            axes[i].legend()
        return fig
