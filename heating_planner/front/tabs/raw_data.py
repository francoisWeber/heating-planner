import streamlit as st
from matplotlib import pyplot as plt

from heating_planner.back.data.base import HazardDataset
from heating_planner.back.data.model.factor import FactorTrend
from heating_planner.back.scoring.factorwise_scoring import FactorsScoringStrategy

N_COLS = 4


@st.fragment
def display():
    ds_proj: HazardDataset = st.session_state.dataset_proj
    ds_ref: HazardDataset = st.session_state.dataset_ref

    factor_scoring_strategy = st.radio(
        "Values to display",
        FactorsScoringStrategy.get_options(),
        index=0,
        key="scoring_method_raw_data",
    )

    scores: HazardDataset = factor_scoring_strategy(
        dataset_proj=ds_proj,
        dataset_hist=ds_ref,
        optimal_ranges=st.session_state.reference_ranges,
    )

    for i, factor in enumerate(scores.factors):
        if i % N_COLS == 0:
            cols = st.columns(N_COLS)
        with cols[i % N_COLS]:
            fig, ax = plt.subplots()
            cmap = "RdYlGn" if factor.trend == FactorTrend.HIGHER_BETTER else "RdYlGn_r"
            scores.df.plot(
                factor.name, ax=ax, alpha=0.8, legend=True, cmap=cmap, markersize=1.5
            )
            st.subheader(factor.name)
            st.write(factor.description + f" ({factor.trend})")
            st.pyplot(fig, use_container_width=False)
