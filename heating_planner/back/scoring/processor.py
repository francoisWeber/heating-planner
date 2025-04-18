import numpy as np 
import geopandas as gpd
from sklearn.preprocessing import MinMaxScaler

from heating_planner.back.data.base import HazardDataset

from heating_planner.back.streamlit_enums import StreamlitReadyEnum

class Contrast(StreamlitReadyEnum):
    DECREASE = "decrease"
    LEAVE = "leave"
    INCREASE = "increase"
    
    def __call__(self, values: np.ndarray) -> np.ndarray:
        contrast_factor = 1
        if self is Contrast.DECREASE:
            contrast_factor = 0.5
        if self is Contrast.INCREASE:
            contrast_factor = 2
            
        return np.power(values, contrast_factor)
    
    
    

def process_score(dataset: HazardDataset, score: np.ndarray, contrast: Contrast) -> gpd.GeoDataFrame:
        
    score = MinMaxScaler().fit_transform(score.reshape(-1, 1))
    score = contrast(score)
    geo_score = dataset.df[["geometry"]].copy()
    geo_score["score"] = score.squeeze()
    return geo_score