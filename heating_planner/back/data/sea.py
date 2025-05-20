import geopandas as gpd
from loguru import logger

from heating_planner.back.data.base import METRIC_CRS, HazardDataset
from heating_planner.back.data.model.factor import (Factor, FactorTrend,
                                                    FactorType)

FACTOR_NAME = "sea_1m_sub"
FACTOR_DESCR = "Zones innondées avec 1m d'élévation"
FACTOR_TREND = FactorTrend.LOWER_BETTER
FACTOR_TYPE = FactorType.BINARY
FACTOR_UNIT = "binary submersion"

MODEL = "sealevelrise.brgm.fr"
SCENARIO = "1m elevation"

MAX_DISTANCE_M = 1_000


class SeaElevationDataset(HazardDataset):
    """Dataset loader for sea elevation data."""

    @classmethod
    def load_from_path(cls, path: str):
        local_path = cls.resolve_path(path)
        df = gpd.read_file(local_path)
        df = df.to_crs(METRIC_CRS)

        return cls(
            path=path,
            df=df,
            model=MODEL,
            scenario=SCENARIO,
            factors=[Factor(name=FACTOR_NAME, description=FACTOR_DESCR, trend=FACTOR_TREND, type=FACTOR_TYPE, unit=FACTOR_UNIT)],
        )

    @staticmethod
    def process_raw_sea_elevation_dataset(sea_elevation_path: str, reference: gpd.GeoDataFrame | HazardDataset, output_path: str):
        """Raw sea elevation shapefile from https://sealevelrise.brgm.fr/slr/ only contains flooded points
        and resolution if far too high comparing to DRIAS dataset. This methods is used to convert and scale it.

        Args:
            sea_elevation_path (str): path to the sea raw elevation shapefile
            reference (gpd.GeoDataFrame | HazardDataset): a reference dataset to use for the scaling
            output_path (str): where to save the processed sea elevation dataset
        """
        local_path = HazardDataset.resolve_path(sea_elevation_path)
        logger.info(f"Reading raw sea elevation dataset from {local_path}")
        gdf = gpd.read_file(local_path).to_crs("EPSG:2154")
        sea_df = gdf[["geometry", "FID_histol"]].rename(columns={"FID_histol": FACTOR_NAME})

        if isinstance(reference, HazardDataset):
            reference = reference.df

        logger.info(f"Scaling sea elevation dataset with {reference.shape[0]} points")
        scaled_sea_df = gpd.sjoin_nearest(reference[["geometry"]], sea_df, how="left", max_distance=MAX_DISTANCE_M).drop(
            columns="index_right"
        )

        logger.info("Casting sea elevation to binary values 'location submerged'")
        scaled_sea_df[FACTOR_NAME] = (1 - scaled_sea_df[FACTOR_NAME].fillna(1)).astype(bool)

        logger.info(f"Saving processed sea elevation dataset to {output_path}")
        scaled_sea_df.to_file(output_path, driver="ESRI Shapefile")
