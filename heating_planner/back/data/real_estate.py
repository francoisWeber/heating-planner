from typing import List

import geopandas as gpd
import pandas as pd

from heating_planner.back.data.base import METRIC_CRS, HazardDataset
from heating_planner.back.data.model.factor import Factor, FactorTrend, FactorType

FACTOR_NAME = "real_estate_prices"
FACTOR_DEF = "Prix moyen sur 2023/24 DVF"
FACTOR_TREND = FactorTrend.LOWER_BETTER
FACTOR_TYPE = FactorType.CONTINUOUS
FACTOR_UNIT = "euros"

MODEL = "DVF"
SCENARIO = "historique"

FRANCE_LAT_MIN = 40
FRANCE_LAT_MAX = 55
FRANCE_LON_MIN = -5
FRANCE_LON_MAX = 9

SJOIN_MAX_DIST_AGG_DVF = 20_000


class RealEstatePricesEuroPerM2(HazardDataset):
    @classmethod
    def load_from_path(cls, path: str):
        df = gpd.read_file(path)
        df = df.to_crs(METRIC_CRS)  # Convert to WGS84

        factors = [Factor(name=FACTOR_NAME, description=FACTOR_DEF, trend=FACTOR_TREND, type=FACTOR_TYPE, unit=FACTOR_UNIT)]

        return cls(
            path=path,
            df=df,
            model=MODEL,
            scenario=SCENARIO,
            factors=factors,
        )

    @staticmethod
    def _filter_dvf_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df[df.type_local.isin(["Maison", "Appartement"])].reset_index(drop=True).dropna(axis=1, how="all")
        df = df[df.nature_mutation.isin(["Vente", "Vente en l'état futur d'achèvement"])].reset_index(drop=True)
        df = df[["latitude", "longitude", "surface_reelle_bati", "type_local", "nom_commune", "valeur_fonciere"]]
        df["price_per_m2"] = df.valeur_fonciere / df.surface_reelle_bati
        df = df[df.price_per_m2 < 1e4].reset_index(drop=True)

    @staticmethod
    def _merge_dvf_data_onto_ref_geometry(dvf_df: gpd.GeoDataFrame, ref_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        gdf = gpd.sjoin_nearest(ref_df.to_crs(METRIC_CRS), dvf_df.to_crs(METRIC_CRS), how="left", max_distance=SJOIN_MAX_DIST_AGG_DVF)
        return gpd.GeoDataFrame(gdf.groupby("geometry").price_per_m2.mean().reset_index(), geometry="geometry").to_crs(
            SJOIN_MAX_DIST_AGG_DVF
        )

    @staticmethod
    def _convert_dvf_to_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326").drop(
            columns=["longitude", "latitude"]
        )

    @staticmethod
    def _filter_dvf_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        gdf = gdf[gdf.geometry.apply(lambda x: x.is_valid)]
        gdf = gdf.cx[FRANCE_LON_MIN:FRANCE_LON_MAX, FRANCE_LAT_MIN:FRANCE_LAT_MAX]
        return gdf

    @staticmethod
    def prepare_dvf_data_wrt_reference_data(dvf_paths: str | List[str], ref_ds: HazardDataset, output_dir: str) -> None:
        if isinstance(dvf_paths, str):
            dvf_paths = [dvf_paths]
        df = pd.concat([pd.read_csv(path) for path in dvf_paths]).reset_index(drop=True)
        df = RealEstatePricesEuroPerM2._filter_dvf_df(df)
        gdf = RealEstatePricesEuroPerM2._convert_dvf_to_gdf(df)
        gdf = RealEstatePricesEuroPerM2._filter_dvf_gdf(gdf)
        gdf = RealEstatePricesEuroPerM2._merge_dvf_data_onto_ref_geometry(gdf, ref_ds.df)
        gdf.to_file(output_dir)
