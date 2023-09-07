import streamlit as st
import click
from heating_planner.streamlit import display

st.set_page_config(layout="wide")


@click.command()
@click.option("--metadata-path", type=str, default=None)
@click.option("--metadata-aux-path", type=str, default=None)
@click.option("--viable-path", type=str, default=None)
@click.option("--mask-path", type=str, default=None)
@click.option("--hypercube-path", type=str, default=None)
def run(metadata_path, metadata_aux_path, viable_path, mask_path, hypercube_path):
    display.display(
        metadata_path, metadata_aux_path, viable_path, mask_path, hypercube_path
    )


if __name__ == "__main__":
    run(standalone_mode=False, auto_envvar_prefix="HEAT")
