import geopandas as gpd

from heating_planner.back.data.base import Factor, FactorTrend, FactorType, HazardDataset

FACTOR_NAME = "overflood"
FACTOR_DESCR = "Zones touchées par la montée des eaux (1m d'élévation)"
FACTOR_TREND = FactorTrend.LOWER_BETTER
FACTOR_TYPE = FactorType.BINARY

MODEL = "sealevelrise.brgm.fr"
SCENARIO = "1m elevation"


class SeaElevationDataset(HazardDataset):
    """Dataset loader for sea elevation data."""

    @classmethod
    def load_from_path(cls, path: str):
        df = gpd.read_file(path)
        df = df.to_crs(epsg=4326)

        return cls(
            path=path,
            df=df,
            model=MODEL,
            scenario=SCENARIO,
            factors=[Factor(name=FACTOR_NAME, description=FACTOR_DESCR, trend=FACTOR_TREND, type=FACTOR_TYPE)],
        )
