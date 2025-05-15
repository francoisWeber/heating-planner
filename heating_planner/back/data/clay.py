import geopandas as gpd

from heating_planner.back.data.base import HazardDataset, METRIC_CRS
from heating_planner.back.data.model.factor import Factor, FactorTrend, FactorType

FACTOR_NAME = "clay_hazard"
FACTOR_DEF = "Niveau de risque de retrait-gonflement des argiles (3 niveaux)"
FACTOR_TREND = FactorTrend.LOWER_BETTER
FACTOR_TYPE = FactorType.DISCRETE
FACTOR_UNIT = "3-level"

MODEL = "georisques"
SCENARIO = "historique"


class ClayHazardDataset(HazardDataset):
    """Dataset loader for data like
    https://www.georisques.gouv.fr/donnees/bases-de-donnees/retrait-gonflement-des-argiles
    """

    @classmethod
    def load_from_path(cls, path: str):
        local_path = cls.resolve_path(path)
        df = gpd.read_file(local_path)
        df = df.drop(columns=["DPT", "ALEA"]).rename(columns={"NIVEAU": FACTOR_NAME})
        df = df.to_crs(METRIC_CRS)  # Convert to WGS84

        factors = [Factor(name=FACTOR_NAME, description=FACTOR_DEF, trend=FACTOR_TREND, type=FACTOR_TYPE, unit=FACTOR_UNIT)]

        return cls(
            path=path,
            df=df,
            model=MODEL,
            scenario=SCENARIO,
            factors=factors,
        )
