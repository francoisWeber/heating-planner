import streamlit as st
import pandas as pd


def maybe_add_to_session_state(key, value, force=False):
    if key not in st.session_state or force:
        st.session_state[key] = value


@st.cache_resource(show_spinner="loading tabular data ...")
def load_json(df_path, **kwargs):
    return pd.read_json(df_path, **kwargs)


def text_center(txt):
    return st.markdown(
        f'<div style="text-align: center;">{txt}</div>', unsafe_allow_html=True
    )
