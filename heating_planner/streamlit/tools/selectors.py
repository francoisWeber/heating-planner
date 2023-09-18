from typing import List
from .utils import maybe_add_to_session_state
import streamlit as st


class Selector:
    def __init__(
        self,
        type: str,
        key,
        label,
        values: List | bool | None = None,
        default=None,
        refers_to_variable: str | None = None,
    ) -> None:
        self.label = label
        self.values = values
        self.default = default if default else 0
        self.key = key
        self.refers_to_variable = refers_to_variable
        self.type = type
        maybe_add_to_session_state(self.key, self.default)

    def get_st_object(self):
        if self.type == "toggle":
            st.toggle(label=self.label, value=self.default, key=self.key)
        elif self.type == "radio":
            st.radio(
                label=self.label,
                options=self.values,
                index=0,
                key=self.key,
            )
        elif self.type == "slider":
            st.select_slider(
                label=self.label,
                options=self.values,
                key=self.key,
                value=self.default,
            )
