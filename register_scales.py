import streamlit as st
import json
import os
from glob import glob
from os import path as osp
import pandas as pd
from PIL import Image
import numpy as np

st.title("Give your run a label !")


INPUT_DIR = "/Users/f.weber/tmp-fweber/heating/"
OUTPUT_PATH = "/Users/f.weber/tmp-fweber/heating/data1.json"


def question_info2streamlit(question: dict):
    if question["type"] == "number":
        el = st.slider(
            question["label"],
            min_value=question["bounds"][0],
            max_value=question["bounds"][1],
            value=question["default"],
            step=1,
            disabled=False,
            label_visibility="visible",
        )
        return el
    elif question["type"] == "bool":
        el = st.checkbox(question["label"], value=True)
    elif question["type"] == "enum":
        el = st.selectbox(question["label"], question["enum"], on_change=get_next_item)
    return el


def gather(questions, st_questions):
    return {question["id"]: answer for question, answer in zip(questions, st_questions)}


def list_files(directory):
    ls = [f for f in glob(directory + "/**/*.png", recursive=True)]
    if st.session_state.previous_data is None:
        return sorted(ls)
    else:
        already_done = {entry["file"] for entry in st.session_state.previous_data}
        st.session_state.idx = len(already_done)
        return list(already_done) + sorted(list(set(ls).difference(already_done)))


def maybe_load_previous_data(path) -> list:
    try:
        return pd.read_json(path, lines=True).to_dict(orient="records")
    except:
        return None


if "idx" not in st.session_state:
    st.session_state.idx = 0

if "previous_data" not in st.session_state:
    st.session_state.previous_data = maybe_load_previous_data(OUTPUT_PATH)

if "files_paths" not in st.session_state:
    st.session_state.files_paths = list_files(INPUT_DIR)


if "current_file" not in st.session_state:
    st.session_state.current_file = None

if "max_id" not in st.session_state:
    st.session_state.max_id = len(st.session_state.files_paths)

if "data_infos" not in st.session_state:
    st.session_state.data_infos = []


if "scale_ticks" not in st.session_state:
    st.session_state.scale_ticks = ""


def save():
    df = pd.DataFrame(data=st.session_state.data_infos)
    df.to_json(OUTPUT_PATH, orient="records", lines=True)


def push():
    st.session_state.data_infos.append(
        {
            "range": [int(e.strip()) for e in st.session_state.scale_ticks.split(",")],
            "file": st.session_state.current_file,
        }
    )


def get_next_item():
    if st.session_state.idx < st.session_state.max_id - 1:
        st.session_state.idx += 1
    else:
        st.write("This was the last")
        save()
        st.stop()


def increase_range():
    st.session_state.range_min_value = st.session_state.range_min_value * 10
    st.session_state.range_max_value = st.session_state.range_max_value * 10


def decrease_range():
    st.session_state.range_min_value = st.session_state.range_min_value // 10
    st.session_state.range_max_value = st.session_state.range_max_value // 10


def on_click():
    push()
    get_next_item()


bar_context, figure_context = st.columns([2, 1])
with bar_context:
    st.progress(st.session_state.idx / st.session_state.max_id)
with figure_context:
    st.text(f"{st.session_state.idx}e element en cours sur {st.session_state.max_id}")

image_context, form_context = st.columns([3, 1])

with st.form("form", clear_on_submit=True):
    st.session_state.current_file = st.session_state.files_paths[st.session_state.idx]
    with image_context:
        st.text(osp.basename(st.session_state.current_file))
        im = Image.open(st.session_state.current_file)
        st.image(im)
    with form_context:
        st.session_state.scale_ticks = st.text_input("scale ticks")
    submitted = st.form_submit_button("go", on_click=on_click)
save_btn = st.button("save", on_click=save)
