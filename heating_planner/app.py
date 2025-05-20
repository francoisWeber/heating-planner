import streamlit as st
import streamlit_authenticator as stauth
import yaml

from heating_planner.front.tabs import (map, parameters, raw_data,
                                        reference_definition)

st.set_page_config(layout="wide", page_title="Heating Planner", page_icon="🌍")

with open("./credentials.yaml") as f:
    config = yaml.load(f, Loader=yaml.loader.SafeLoader)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

authenticator.logout("logout")

name, logged_status, username = authenticator.login("Login id/creds")

if logged_status:
    tabs = st.tabs(["Map", "Params", "Reference definition", "raw data"])

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

        with tabs[3]:
            raw_data.display()

elif logged_status is False:
    st.error("Wrong credentials !", icon="🚨")
else:
    pass
