from heating_planner.back.geo import geo_tool
import geopandas as gpd
from shapely.geometry import Point

class HazardDataset:
    def __init__(self, path: str | None = None, df: gpd.GeoDataFrame | None = None, columns_definition: dict | None = None, model: str | None = None, scenario: str | None = None, is_boolean: bool = False):
        self.path = path
        self.df = df
        self.columns_definition = columns_definition
        self.model = model
        self.scenario = scenario
        self.is_boolean = is_boolean
    
    @classmethod
    def load_from_path(cls, path: str):
        """Load dataset from path"""
        raise NotImplementedError("Must be implemented in subclass")
    

    def get_index_of_city(self, city: str) -> int:
        loc = geo_tool.geocode(city)
        point = Point(loc.longitude, loc.latitude)
        return self.df['geometry'].distance(point).idxmin()
    
    def __add__(self, other: "HazardDataset"):
        if not isinstance(other, HazardDataset):
            raise TypeError("Can only add two HazardDataset objects")
        if self.df is None or other.df is None:
            raise ValueError("Cannot add datasets with None df")
        
        merged_df = gpd.sjoin_nearest(self.df, other.df, how="inner").drop(columns=["index_right"])
        path = f"{self.path}_{other.path}"
        columns_definition = {**self.columns_definition, **other.columns_definition}
        model = f"{self.model}_{other.model}"
        scenario = f"{self.scenario}_{other.scenario}"
        return HazardDataset(df=merged_df, path=path, columns_definition=columns_definition, model=model, scenario=scenario)
    
    def __radd__(self, other: "HazardDataset"):
        if other == None:
            return self
        else:
            return self.__add__(other)