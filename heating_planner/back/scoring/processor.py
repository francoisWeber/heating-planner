from typing import Dict

import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from loguru import logger

from heating_planner.back.data.base import HazardDataset
from heating_planner.back.data.model.factor import Factor
from heating_planner.back.geo.tools import make_geo_df
from heating_planner.back.scoring.scaler import ScoreScalingStrategy
from heating_planner.back.streamlit_enums import StreamlitReadyEnum

SCORE_COLNAME = "score"


class Contrast(StreamlitReadyEnum):
    DECREASE = "decrease"
    LEAVE = "leave"
    INCREASE = "increase"

    def call_on_df(self, df: pd.DataFrame) -> pd.DataFrame:
        contrast_factor = 1
        if self is Contrast.DECREASE:
            contrast_factor = 0.5
        if self is Contrast.INCREASE:
            contrast_factor = 2

        df[SCORE_COLNAME] = np.power(df[SCORE_COLNAME].values, contrast_factor)

        return df


def combine_binary_masks(binary_dfs: pd.DataFrame) -> pd.DataFrame:
    mask = np.prod(binary_dfs.to_numpy(dtype=int), axis=1)
    return pd.DataFrame(mask, index=binary_dfs.index, columns=["mask"])


def process_score(
    dataset: HazardDataset,
    contrast: Contrast,
    binary_factor_infos: Dict[Factor, bool],
    scaling_strategy: ScoreScalingStrategy = ScoreScalingStrategy.MINMAX,
) -> gpd.GeoDataFrame:
    df, geometry, _, _ = dataset.explode_information()
    assert SCORE_COLNAME in df, f"No column {SCORE_COLNAME} in data !"

    final_df = scaling_strategy.call_on_df(df)
    final_df = contrast.call_on_df(final_df)
    final_df = make_geo_df(final_df, geometry=geometry)

    binary_factors = [factor.name for factor, active in binary_factor_infos.items() if active]
    if binary_factors:
        logger.info(f"Processing binary factors: {binary_factors}")
        binary_dfs = df[binary_factors]
        mask = combine_binary_masks(binary_dfs)
        penalty = make_geo_df(mask, geometry)
        final_df = gpd.overlay(final_df, penalty)
        final_df[SCORE_COLNAME] = np.where(final_df["mask"], final_df[SCORE_COLNAME], 0)

    return final_df
