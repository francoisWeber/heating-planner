import streamlit as st
from heating_planner.front.cst import VAR_TREND_2_EMOJI
from heating_planner.front.histogram import InteractiveHistogram
import json

DEFAULT_REF_CITIES = "Strasbourg, L'Aigle, Morlaix"

PARAMS_INIT = {"reference_ranges": {}}


def display():
    st.subheader("Reference definition")
    if not st.session_state.loaded:
        st.warning("Please load the files first")
        st.stop()
    else:
        dataset_proj = st.session_state.dataset_proj
        dataset_ref = st.session_state.dataset_ref

        ref_keys = set(dataset_ref.factors_definitions.keys())
        proj_keys = set(dataset_proj.factors_definitions.keys())
        common_keys = sorted(list(ref_keys.intersection(proj_keys)))

        cols = st.columns(3)
        with cols[0]:
            ref_cities = st.text_input("Cities", value=DEFAULT_REF_CITIES)
        with cols[1]:
            st.download_button(
                label="Download reference ranges",
                data=json.dumps(st.session_state.reference_ranges, indent=4),
                file_name="reference_ranges.json",
                mime="application/json",
            )
        ref_cities = [city.strip() for city in ref_cities.split(",")]
        ref_indices = [dataset_ref.get_index_of_city(city) for city in ref_cities]
        ref_values = {key: {city: dataset_ref.df.iloc[index][key] for city, index in zip(ref_cities, ref_indices)} for key in common_keys}

        NCOL = 3
        cols = st.columns(NCOL)
        for i, key in enumerate(common_keys):
            with cols[i % NCOL]:
                st.subheader(key)
                var_definition = dataset_proj.factors_definitions[key]
                var_trend = VAR_TREND_2_EMOJI[dataset_proj.factors_types[key]]
                st.write(var_definition + " " + var_trend)
                hist_name = f"chart-{key}"
                if hist_name not in st.session_state:
                    st.session_state[hist_name] = InteractiveHistogram(dataset_ref.df[key], key, ref_values[key])
                histo: InteractiveHistogram = st.session_state[hist_name]
                slider_args = histo.get_slider_args()
                min_v, max_v = st.slider(**slider_args)
                chart = histo.alter_chart_between_range(min_v, max_v)
                st.altair_chart(chart, use_container_width=True)
                st.session_state.reference_ranges[key] = (min_v, max_v)
