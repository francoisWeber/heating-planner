from typing import List
from .utils import maybe_add_to_session_state
import streamlit as st


class Selector:
    def __init__(
        self,
        key,
        label,
        values: List | bool | None = None,
        default=None,
        is_toggle=False,
        refers_to_variable: str | None = None,
    ) -> None:
        self.label = label
        if values is None and not is_toggle:
            raise ValueError("Values must be set")
        self.values = values
        self.default = default if default else 0
        self.key = key
        self.is_toggle = is_toggle
        self.refers_to_variable = refers_to_variable
        maybe_add_to_session_state(self.key, self.default)

    def get_st_object(self):
        if self.is_toggle:
            st.toggle(label=self.label, value=self.default, key=self.key)
        else:
            try:
                st.radio(
                    label=self.label,
                    options=self.values,
                    index=0,
                    key=self.key,
                )
            except:
                st.text(self.label)
