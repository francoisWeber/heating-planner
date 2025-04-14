from typing import Dict
import pandas as pd
import altair as alt
import math as m


class InteractiveHistogram:
    def __init__(self, data: pd.Series, title: str, reference_values: Dict[str, float], nbins=50):
        self.data = data.to_frame()
        self.var = data.name
        self.title = title
        self.ref = pd.DataFrame([{"label": key, self.var: val} for key, val in reference_values.items()])
        self.min_val = m.floor(data.min())
        self.max_val = m.ceil(data.max())
        self.nbins = nbins
        self.chart = self.create_altair_base_chart()

    def get_slider_args(self):
        slider_kwargs = {
            "label": f"Histogram for {self.var}",
            "min_value": self.min_val,
            "max_value": self.max_val,
            "value": (self.min_val, self.max_val),
        }

        return slider_kwargs

    def create_altair_base_chart(self):
        chart = (
            alt.Chart(self.data)
            .transform_bin("binned_value", field=self.var, bin=alt.Bin(maxbins=self.nbins))
            .transform_aggregate(count="count()", groupby=["binned_value"])
        )

        # Create vertical rules with text labels

        return chart  # + ref_lines + ref_labels

    def alter_chart_between_range(self, min_val, max_val, width=600, height=400):
        chart = (
            self.chart.transform_calculate(highlight=f"{min_val} <= datum.binned_value && datum.binned_value < {max_val}")
            .mark_bar()
            .encode(
                x=alt.X("binned_value:Q", title=self.var),
                y=alt.Y("count:Q", title="Count", axis=None),
                color=alt.condition("datum.highlight", alt.value("orange"), alt.value("lightgray")),
            )
            .properties(width=width, height=height, title=self.title)
        )

        ref_lines = (
            alt.Chart(self.ref)
            .mark_rule(strokeWidth=2)
            .encode(
                x=f"{self.var}:Q",
                color=alt.Color("label:N", legend=None),  # Assign color based on the label
            )
        )
        ref_labels = (
            alt.Chart(self.ref)
            .mark_text(align="left", dy=-5, dx=2, fontSize=16)
            .encode(
                x=f"{self.var}:Q",
                y=alt.Y("row_number:O", title=None, axis=None, sort="descending"),  # Stagger labels vertically
                text="label:N",
                color=alt.Color("label:N", legend=None),  # Assign color based on the label
            )
            .transform_window(row_number="row_number()", sort=[alt.SortField(f"{self.var}:Q", order="ascending")])
        )
        return chart + ref_lines + ref_labels
