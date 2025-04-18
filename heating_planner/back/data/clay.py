import geopandas as gpd

from heating_planner.back.data.base import DatasetFactors, FactorType, HazardDataset, Factor

FACTOR_NAME = "clay_hazard"
FACTOR_DEF = "Niveau de risque de retrait-gonflement des argiles (3 niveaux)"
FACTOR_TYPE = FactorType.LOWER_BETTER

MODEL = "georisques"
SCENARIO = "historique"
VAR_NAME = "clay_hazard"


class ClayHazardDataset(HazardDataset):
    """Dataset loader for data like
    https://www.georisques.gouv.fr/donnees/bases-de-donnees/retrait-gonflement-des-argiles
    """

    @classmethod
    def load_from_path(cls, path: str):
        df = gpd.read_file(path)
        df = df.drop(columns=["DPT", "ALEA"]).rename(columns={"NIVEAU": VAR_NAME})
        df = df.to_crs(epsg=4326)  # Convert to WGS84

        factors = DatasetFactors([Factor(name=VAR_NAME, description=FACTOR_DEF, type=FACTOR_TYPE)])

        return cls(
            path=path,
            df=df,
            model=MODEL,
            scenario=SCENARIO,
            factors=factors,
        )
