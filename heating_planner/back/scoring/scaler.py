import geopandas as gpd
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from heating_planner.back.data.base import HazardDataset
from heating_planner.back.streamlit_enums import StreamlitReadyEnum


class ScoreScalingStrategy(StreamlitReadyEnum):
    """Apply a scaling strategy to a GeoDataFrame preserving the geometry column."""

    MINMAX = "min max"
    STANDARD = "standard"
    RANK = "rank"
    NONE = "none"

    def __call__(self, dataset: HazardDataset) -> HazardDataset:
        if self is ScoreScalingStrategy.NONE:
            return dataset

        df, geometry, factors, _ = dataset.explode_information()
        scaled_df = self.call_on_df(df)
        return HazardDataset(
            df=gpd.GeoDataFrame(pd.concat([geometry, scaled_df], axis=1)),
            factors=factors,
        )

    def call_on_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the selected scaling strategy to the dataframe."""
        if self is ScoreScalingStrategy.RANK:
            return df.rank(method="min", pct=True)
        elif self is ScoreScalingStrategy.MINMAX:
            scaler = MinMaxScaler()
            return pd.DataFrame(
                scaler.fit_transform(df), index=df.index, columns=df.columns
            )
        elif self is ScoreScalingStrategy.STANDARD:
            scaler = StandardScaler()
            return pd.DataFrame(
                scaler.fit_transform(df), index=df.index, columns=df.columns
            )
        elif self is ScoreScalingStrategy.NONE:
            return df
        else:
            raise ValueError(f"Unknown scaling strategy: {self}")
