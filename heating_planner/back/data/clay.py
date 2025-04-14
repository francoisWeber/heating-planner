import geopandas as gpd

from heating_planner.back.data.base import LOWER_BETTER, HazardDataset

COLUMN_DEFINITION = {"clay_hazard": "Niveau de risque de retrait-gonflement des argiles (3 niveaux)"}
MODEL = "georisques"
SCENARIO = "historique"
VAR_NAME = "clay_hazard"

class ClayHazardDataset(HazardDataset):
    """Dataset loader for data like 
    https://www.georisques.gouv.fr/donnees/bases-de-donnees/retrait-gonflement-des-argiles
    """
    def __init__(self, path: str, df: gpd.GeoDataFrame | None = None):
        trend_preferences = {VAR_NAME: LOWER_BETTER}
        super().__init__(path=path, df=df, columns_definition=COLUMN_DEFINITION, model=MODEL, scenario=SCENARIO, trend_preferences=trend_preferences)

    @classmethod
    def load_from_path(cls, path: str):
        df = gpd.read_file(path)
        df = df.drop(columns=['DPT', 'ALEA']).rename(columns={"NIVEAU": VAR_NAME})
        df = df.to_crs(epsg=4326)  # Convert to WGS84
        
        return cls(
            path=path,
            df=df,
        )
        