import geopandas as gpd
import pandas as pd
from heating_planner.crs import METRIC_CRS

DEFAULT_SCORE_COLNAME = "score"
DEFAULT_SURROUNDING_MAX_DISTANCE = 16_000


def make_geo_df(df: pd.DataFrame, geometry: gpd.GeoSeries) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(pd.concat([geometry, df], axis=1), geometry="geometry")


def get_topn_with_surroundings(
    gdf: gpd.GeoDataFrame,
    n: int,
    score_colname: str = DEFAULT_SCORE_COLNAME,
    surrounding_max_distance: float = DEFAULT_SURROUNDING_MAX_DISTANCE,
    crs: str = METRIC_CRS,
) -> gpd.GeoDataFrame:
    _gdf = gdf.copy().to_crs(METRIC_CRS)
    top_n = _get_topn_with_surroundings_rec(
        _gdf, n, score_colname=score_colname, surrounding_max_distance=surrounding_max_distance, crs=crs
    )
    return gpd.GeoDataFrame(top_n, geometry="geometry").to_crs(gdf.crs)


def _get_topn_with_surroundings_rec(
    gdf: gpd.GeoDataFrame,
    n: int,
    score_colname: str = DEFAULT_SCORE_COLNAME,
    surrounding_max_distance: float = DEFAULT_SURROUNDING_MAX_DISTANCE,
    crs: str = METRIC_CRS,
) -> pd.DataFrame:
    top1 = gdf.nlargest(1, score_colname)

    if n == 1:
        return top1
    else:
        top1_surroundings = gdf.sjoin_nearest(top1, how="inner", max_distance=surrounding_max_distance)
        gdf.loc[top1_surroundings.index, score_colname] = 0
        return pd.concat(
            [
                top1,
                get_topn_with_surroundings(
                    gdf, n - 1, score_colname=score_colname, surrounding_max_distance=surrounding_max_distance, crs=crs
                ),
            ]
        )
