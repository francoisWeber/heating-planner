import streamlit as st

from heating_planner.front.tabs import map, parameters, reference_definition

st.set_page_config(layout="wide")

tabs = st.tabs(["Map", "Params", "Reference definition"])

PARAMS_INIT = {**parameters.PARAMS_INIT, **reference_definition.PARAMS_INIT, **map.PARAMS_INIT}
for key, value in PARAMS_INIT.items():
    if key not in st.session_state:
        st.session_state[key] = value


with tabs[1]:
    parameters.display()

if st.session_state.loaded:
    with tabs[0]:
        map.display()

    with tabs[2]:
        reference_definition.display()
