from typing import List
import streamlit as st

from heating_planner.front.loading import load_and_cache_dataset
from heating_planner.back.data import ClayHazardDataset, DriasDataset, SeaElevationDataset
from heating_planner.back.data.base import HazardDataset
from heating_planner.back.streamlit_enums import StreamlitReadyEnum


class HazardDatasetType(StreamlitReadyEnum):
    DRIAS = "DRIAS dataset"
    SEA = "Sea elevation dataset"
    CLAY = "Clay variation dataset"

    def get_class(self) -> HazardDataset:
        if self is HazardDatasetType.DRIAS:
            return DriasDataset
        if self is HazardDatasetType.SEA:
            return SeaElevationDataset
        if self is HazardDatasetType.CLAY:
            return ClayHazardDataset
        raise ValueError()


PARAMS_INIT = {
    "loaded": False,
    "dataset_ref": None,
    "dataset_proj": None,
}

drias_ref_path = "/Users/francois.weber/perso/tmp/heating2/data/drias/ref_indicesMAX_25041416092929830.txt"
drias_proj_path = "https://cloud.fweber.fr/s/Tx7TFWMJxBk5WtB/download"
clay_path = "/Users/francois.weber/perso/tmp/heating2/data/clay/drias_clay.shp"
sea_elevation_path = "/Users/francois.weber/perso/tmp/heating2/data/sea_elevation/sea_elevation.shp"


@st.fragment
def display():
    st.header("Parameters")

    ref_datasets: List[HazardDataset] = []
    proj_datasets: List[HazardDataset] = []

    reference_col, projection_col = st.columns(2)
    with reference_col:
        st.subheader("Indicate what dataset to use as reference")
        with st.container(border=True):
            data_loader = st.selectbox("Dataset type", HazardDatasetType.get_available_options(), index=0, key="params-select-ref-1")
            dataset_path = st.text_input("Dataset path", value=drias_ref_path, key="params-txt-input-ref-1")
            ref_datasets.append(load_and_cache_dataset(dataset_path, data_loader.get_class()))

        with st.container(border=True):
            data_loader = st.selectbox("Dataset type", HazardDatasetType.get_available_options(), index=1, key="params-select-ref-2")
            dataset_path = st.text_input("Dataset path", value=sea_elevation_path, key="params-txt-input-ref-2")
            ref_datasets.append(load_and_cache_dataset(dataset_path, data_loader.get_class()))

        with st.container(border=True):
            data_loader = st.selectbox("Dataset type", HazardDatasetType.get_available_options(), index=2, key="params-select-ref-3")
            dataset_path = st.text_input("Dataset path", value=clay_path, key="params-txt-input-ref-3")
            ref_datasets.append(load_and_cache_dataset(dataset_path, data_loader.get_class()))

    with projection_col:
        st.subheader("Indicate what dataset to use as reference")
        with st.container(border=True):
            data_loader = st.selectbox("Dataset type", HazardDatasetType.get_available_options(), index=0, key="params-select-proj-1")
            dataset_path = st.text_input("Dataset path", value=drias_proj_path, key="params-txt-input-proj-1")
            proj_datasets.append(load_and_cache_dataset(dataset_path, data_loader.get_class()))

        with st.container(border=True):
            data_loader = st.selectbox("Dataset type", HazardDatasetType.get_available_options(), index=1, key="params-select-proj-2")
            dataset_path = st.text_input("Dataset path", value=sea_elevation_path, key="params-txt-input-proj-2")
            proj_datasets.append(load_and_cache_dataset(dataset_path, data_loader.get_class()))

        with st.container(border=True):
            data_loader = st.selectbox("Dataset type", HazardDatasetType.get_available_options(), index=2, key="params-select-proj-3")
            dataset_path = st.text_input("Dataset path", value=clay_path, key="params-txt-input-proj-3")
            proj_datasets.append(load_and_cache_dataset(dataset_path, data_loader.get_class()))

    if ref_datasets:
        dataset_ref = sum(ref_datasets[1:], ref_datasets[0])
    else:
        dataset_ref = None
    if proj_datasets:
        dataset_proj = sum(proj_datasets[1:], proj_datasets[0])
    else:
        dataset_proj = None

    st.session_state["dataset_proj"] = dataset_proj
    st.session_state["dataset_ref"] = dataset_ref
    st.session_state["loaded"] = True
    st.markdown(":heavy_check_mark: Files loaded")
