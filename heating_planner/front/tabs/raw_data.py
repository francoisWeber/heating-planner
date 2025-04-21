from matplotlib import pyplot as plt
import streamlit as st 
from heating_planner.back.data.base import HazardDataset, FactorTrend
import geopandas as gpd


N_COLS = 4

def display():
    hazard_ds: HazardDataset = st.session_state.dataset_proj
    df: gpd.GeoDataFrame = hazard_ds.df
    
    for i, factor in enumerate(hazard_ds.factors):
        if i % N_COLS == 0:
            cols = st.columns(N_COLS)
        with cols[i % N_COLS]:
            fig, ax = plt.subplots()
            cmap = "RdYlGn" if factor.trend == FactorTrend.HIGHER_BETTER else "RdYlGn_r"
            df.plot(factor.name, ax=ax, alpha=0.8,legend=True, cmap=cmap)
            st.subheader(factor.name)
            st.write(factor.description + f" ({factor.trend})")
            st.pyplot(fig, use_container_width=False)
    