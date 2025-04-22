import geopandas as gpd
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from heating_planner.back.streamlit_enums import StreamlitReadyEnum


class GeoPandasScalingStrategy(StreamlitReadyEnum):
    """Apply a scaling strategy to a GeoDataFrame preserving the geometry column."""

    MINMAX = "min max"
    STANDARD = "standard"
    RANK = "rank"

    def __call__(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        geometry = df.pop("geometry").to_frame("geometry")
        if self is GeoPandasScalingStrategy.RANK:
            scores = df.rank(method="min", pct=True)
        else:
            scaler = MinMaxScaler() if self is GeoPandasScalingStrategy.MINMAX else StandardScaler()
            scores = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
        return gpd.GeoDataFrame(pd.concat([geometry, scores], axis=1))
