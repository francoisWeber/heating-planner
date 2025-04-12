import streamlit as st

from heating_planner.front.histogram import InteractiveHistogram
from heating_planner.front.loading import load_drias_dataset
from heating_planner.front.tabs import parameters

st.set_page_config(layout="wide")

tabs = st.tabs(["Params", "Reference definition", "Map"])

PARAMS_INIT = {
    **parameters.PARAMS_INIT,
}
for key, value in PARAMS_INIT.items():
    if key not in st.session_state:
        st.session_state[key] = value

with tabs[0]:
    parameters.display()
    
with tabs[1]:
    st.subheader("Reference definition")
    if st.session_state.loaded:
    
        dataset_proj = st.session_state.dataset_proj
        dataset_ref = st.session_state.dataset_ref
        
        ref_keys = set(dataset_ref.columns_definition.keys())
        proj_keys = set(dataset_proj.columns_definition.keys())
        common_keys = sorted(list(ref_keys.intersection(proj_keys)))
        
        ref_cities = st.text_input("Cities", "Strasbourg, L'Aigle, Morlaix")
        ref_cities = [city.strip() for city in ref_cities.split(",")]
        ref_indices = [dataset_ref.get_index_of_city(city) for city in ref_cities]
        ref_values = {key: {city: dataset_ref.df.iloc[index][key] for city, index in zip(ref_cities, ref_indices)} for key in common_keys}




        NCOL = 3
        cols = st.columns(NCOL)
        for i, key in enumerate(common_keys):
            with cols[i % NCOL]:
                st.subheader(key)
                st.write(dataset_proj.columns_definition[key])
                hist_name = f"chart-{key}"
                if hist_name not in st.session_state:
                    st.session_state[hist_name] = InteractiveHistogram(dataset_ref.df[key], key, ref_values[key])
                histo: InteractiveHistogram = st.session_state[hist_name]
                slider_args = histo.get_slider_args()
                min_v, max_v = st.slider(**slider_args)
                chart = histo.alter_chart_between_range(min_v, max_v)
                st.altair_chart(chart, use_container_width=True)
                
    else:
        st.warning("Please load the files first")
        st.stop()