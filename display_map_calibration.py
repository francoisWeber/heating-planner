import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from PIL import Image, ImageDraw

st.set_page_config(layout="wide")


def get_ellipse_coords(point: tuple[int, int]) -> tuple[int, int, int, int]:
    center = point
    radius = 10
    return (
        center[0] - radius,
        center[1] - radius,
        center[0] + radius,
        center[1] + radius,
    )


locations_names = [
    "Paris",
    "Belfort",
    "Rouen",
    "Rennes",
    "Le Mans",
    "Troyes",
    "Poitiers",
    "Limoges",
    "Villeurbanne",
    "Grenoble",
    "Bordeaux",
    "Toulouse",
    "Montpellier",
    "Avignon",
    "Perpignan",
]


def run(
    basemap_path,
):
    if "points" not in st.session_state:
        st.session_state["points"] = []

    if "idx" not in st.session_state:
        st.session_state["idx"] = 0

    with Image.open(basemap_path) as img:
        draw = ImageDraw.Draw(img)

        # Draw an ellipse at each coordinate in points
        for point in st.session_state["points"]:
            coords = get_ellipse_coords(point)
            draw.ellipse(coords, fill="red")

        st.text("Pointer " + locations_names[st.session_state.idx])
        value = streamlit_image_coordinates(img, key="pil")

        if value is not None:
            point = value["x"], value["y"]
            st.text(point)
            if point not in st.session_state["points"]:
                st.session_state["points"].append(point)
                st.session_state.idx += 1
                if len(st.session_state.points) == len(locations_names):
                    loc2xy = {
                        loc: point
                        for loc, point in zip(locations_names, st.session_state.points)
                    }
                    import json

                    with open(
                        "/Users/f.weber/tmp-fweber/heating/aux_maps/ref_carte_name2xy.json",
                        "w",
                    ) as f:
                        json.dump(loc2xy, f)
                    st.stop()
                # st.experimental_rerun()


if __name__ == "__main__":
    run("/Users/f.weber/tmp-fweber/heating/aux_maps/ref_carte_drias_f1.png")
