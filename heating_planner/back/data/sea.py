import geopandas as gpd
from heating_planner.back.data.base import FactorType, HazardDataset

COLUMN_DEFINITION = {"overflood": "Zones touchées par la montée des eaux (1m d'élévation)"}
MODEL = "sealevelrise.brgm.fr"
SCENARIO = "1m elevation"

class SeaElevationDataset(HazardDataset):
    """Dataset loader for sea elevation data.
    """
    def __init__(self, path: str, df: HazardDataset | None = None):
        trend_preferences = {"overflood": FactorType.BINARY}
        super().__init__(path=path, df=df, factors_definitions=COLUMN_DEFINITION, model=MODEL, scenario=SCENARIO, is_boolean=True, factors_types=trend_preferences)
        
    @classmethod
    def load_from_path(cls, path: str):
        df = gpd.read_file(path)
        df = df.to_crs(epsg=4326)
        return cls(
            path=path,
            df=df,
        )