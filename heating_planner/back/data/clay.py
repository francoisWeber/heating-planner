import geopandas as gpd

from heating_planner.back.data.base import HazardDataset

COLUMN_DEFINITION = {"clay_hazard": "Niveau de risque de retrait-gonflement des argiles (3 niveaux)"}
MODEL = "georisques"
SCENARIO = "historique"

class ClayHazardDataset(HazardDataset):
    """Dataset loader for data like 
    https://www.georisques.gouv.fr/donnees/bases-de-donnees/retrait-gonflement-des-argiles
    """
    def __init__(self, path: str, df: gpd.GeoDataFrame | None = None):
        super().__init__(path=path, df=df, columns_definition=COLUMN_DEFINITION, model=MODEL, scenario=SCENARIO)

    @classmethod
    def load_from_path(cls, path: str):
        df = gpd.read_file(path)
        df = df.drop(columns=['DPT', 'ALEA']).rename(columns={"NIVEAU": "clay_hazard"})
        df = df.to_crs(epsg=4326)  # Convert to WGS84
        
        return cls(
            path=path,
            df=df,
        )
        