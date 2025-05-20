from typing import List, Tuple
import streamlit as st
from matplotlib import pyplot as plt
import geopandas as gpd

from heating_planner.back.data.base import HazardDataset
from heating_planner.back.data.model.factor import Factor
from heating_planner.back.geo.coder import geocoding
from heating_planner.back.geo.tools import get_topn_with_surroundings
from heating_planner.back.scoring.factorwise_scoring import FactorsScoringStrategy
from heating_planner.back.scoring.fusion import ScoringFusion
from heating_planner.back.scoring.processor import Contrast, apply_binary_masks, process_score
from heating_planner.back.scoring.scaler import ScoreScalingStrategy
from heating_planner.crs import GPS_CRS

PARAMS_INIT = {}
MARKER_SIZE = 2.0

FACTOR_WEIGHTS_NCOLS = 3


class OptionsDisplayer:

    @staticmethod
    def display():
        scores_col, scaling_col, display_col = st.columns(3)
        with scores_col:
            st.subheader("Factorwise scoring strategy")
            factors_scoring_strategy = OptionsDisplayer.display_factors_scoring_strategy()
        with scaling_col:
            st.subheader("Factor's scores fusion")
            factors_scaling = OptionsDisplayer.display_factors_scaling()
        with display_col:
            st.subheader("Display mode")
            contrast = OptionsDisplayer.display_contrast()
        fusion_strategy = OptionsDisplayer.display_fusion_strategy()
        return factors_scoring_strategy, factors_scaling, fusion_strategy, contrast

    @staticmethod
    def display_factors_scoring_strategy():
        return st.radio("scoring method", FactorsScoringStrategy.get_options(), index=0, key="scoring_method")

    @staticmethod
    def display_factors_scaling():
        return st.radio("factors score scaling", ScoreScalingStrategy.get_options())

    @staticmethod
    def display_fusion_strategy():
        return ScoringFusion.WEIGHTED_MEAN

    @staticmethod
    def display_contrast():
        return st.radio("contrast management", options=Contrast.get_options(), index=1)

class WeightsDisplayer:
    @staticmethod
    def display(factors_continous: List[Factor], factors_binary: List[Factor]):
        factors_continuous_coefs = {factor: 1 for factor in factors_continous}
        factors_binary_activation = {factor: 1 for factor in factors_binary}
        with st.container(border=True):
            for i, factor in enumerate(factors_binary):
                factors_binary_activation[factor] = st.toggle("With: " + factor.description, value=True)

            factors_cols = st.columns(FACTOR_WEIGHTS_NCOLS)
            with st.form("weight-form"):
                # display weightable factors
                for i, factor in enumerate(factors_continous):
                    if i % FACTOR_WEIGHTS_NCOLS == 0:
                        factors_cols = st.columns(FACTOR_WEIGHTS_NCOLS)

                    with factors_cols[i % FACTOR_WEIGHTS_NCOLS]:
                        factors_continuous_coefs[factor] = st.slider(factor.description[:50], 0, 3, step=1, value=factors_continuous_coefs[factor])
                st.form_submit_button("update")
                
        return factors_continuous_coefs, factors_binary_activation
    
class MapDisplayer:
    @staticmethod
    def display(ds_hist_continuous, ds_hist_binary, ds_proj_continuous, factor_scoring_strategy, factors_scaling, fusion_strategy, contrast, factors_continuous_coefs, factors_binary_activation) -> Tuple[gpd.GeoDataFrame, HazardDataset]:
        ds_scored_factors = factor_scoring_strategy(ds_proj_continuous, ds_hist_continuous, st.session_state.reference_ranges)
        factorwise_scores = factors_scaling(ds_scored_factors)
        fused_score = fusion_strategy(factorwise_scores, factors_continuous_coefs)
        # Display only rows where score is not a float
        fused_score = process_score(fused_score, contrast=contrast)
        fused_score = apply_binary_masks(fused_score, factors_binary_activation, ds_hist_binary)
        fig, ax = plt.subplots()
        fused_score.plot("score", ax=ax, legend=True, cmap="RdYlGn", markersize=MARKER_SIZE)
        st.pyplot(fig)
        st.download_button("Download GeoDF", fused_score.to_json(), "heating_map_scores.json")
        
        return fused_score, factorwise_scores
    
class TopCitiesDisplay:
    @staticmethod
    def display(processed_scores):
        st.subheader("Top 10 points")
        top_rows = get_topn_with_surroundings(processed_scores, n=10, score_colname="score").to_crs(GPS_CRS)
        geometries = top_rows.geometry.to_list()
        coords = [tuple(coord[0] for coord in geo.coords.xy[::-1]) for geo in geometries]
        locations = [geocoding.reverse(coord) for coord in coords]
        for i, loc in enumerate(locations):
            st.markdown(f"**Top {i + 1}**\n => {loc.address}")
            
class CitiesAnalysisDisplayer:
    @staticmethod
    def display(processed_scores: gpd.GeoDataFrame, factorwise_scores: HazardDataset):
        st.subheader("Score of selected cities")
        cities_str = st.text_input("cities to check")
        cities = [city.strip() for city in cities_str.split(",") if len(city.strip()) >= 3]
        if cities:
            index_of_cities = [factorwise_scores.get_index_of_city(city) for city in cities]
            scores_of_cities = processed_scores.iloc[index_of_cities]
            scores_of_cities = scores_of_cities.drop(columns="geometry").assign(city=cities)
            st.dataframe(scores_of_cities)
            st.dataframe(factorwise_scores.df.iloc[index_of_cities].drop(columns="geometry").assign(city=cities))
        

@st.fragment
def display():
    if not st.session_state.loaded:
        st.warning("Please load the files first")
        st.button("retry")
    else:
        _, ds_proj_continuous = st.session_state.dataset_proj.split_by_factor_type()
        ds_hist_binary, ds_hist_continuous = st.session_state.dataset_ref.split_by_factor_type()
        
        factors_continous = sorted(list(set(ds_hist_continuous.factors).intersection(ds_proj_continuous.factors)))
        factors_binary = ds_hist_binary.factors

        factor_scoring_strategy, factors_scaling, fusion_strategy, contrast = OptionsDisplayer.display()

        map_and_factors_cols = st.columns(2)
        with map_and_factors_cols[0]:
            factors_continuous_coefs, factors_binary_activation = WeightsDisplayer.display(factors_continous, factors_binary)
        with map_and_factors_cols[1]:
            processed_scores, factorwise_scores = MapDisplayer.display(ds_hist_continuous, ds_hist_binary, ds_proj_continuous, factor_scoring_strategy, factors_scaling, fusion_strategy, contrast, factors_continuous_coefs, factors_binary_activation)

        with st.container(border=True):
            cols = st.columns(2)
            with cols[0]:
                TopCitiesDisplay.display(processed_scores)
            with cols[1]:
                CitiesAnalysisDisplayer.display(processed_scores, factorwise_scores)