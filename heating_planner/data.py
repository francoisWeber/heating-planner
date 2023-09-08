from typing import Tuple
import numpy as np
import pandas as pd
from heating_planner.utils import load_np_from_anywhere, load_pil_from_anywhere


class Datum:
    def __init__(
        self,
        metadata_path,
        metadata_aux_path,
        viable_ranges_path,
        mask_path,
        hypercubes_path,
        base_map_path,
    ) -> None:
        self.metadata = (
            pd.read_json(metadata_path)
            .reset_index()
            .sort_values(["variable", "season"])
        )
        self.metadata_aux = (
            pd.read_json(metadata_aux_path)
            .reset_index()
            .sort_values(["variable", "season"])
        )
        self.df_viable_ranges = pd.read_json(viable_ranges_path)
        self.mask = load_np_from_anywhere(mask_path)
        hypercubes = load_np_from_anywhere(hypercubes_path)
        self.hypercube = hypercubes["map"]
        self.hypercube_aux = hypercubes["aux"]
        self.base_map = load_pil_from_anywhere(base_map_path)

    def get_metadata_for(self, variable, season=None, term=None):
        if variable in self.metadata.variable.unique():
            _df = self.metadata.assign(source="metadata")
        elif variable in self.metadata_aux.variable.unique():
            _df = self.metadata_aux.assign(source="aux")
        else:
            raise ValueError(f"{variable=} found nowhere")
        question = locals()
        _ = question.pop("self")
        _df = _df[_df.variable == variable]
        if len(_df) == 0:
            raise ValueError(f"No data matching {question=}")
        if len(_df) == 1:
            return _df
        else:
            return _df[(_df.season == season) & (_df.term == term)]

    def get_slice_for(self, variable, season=None, term=None):
        metadata = self.get_metadata_for(variable, season, term)
        if metadata["source"].iloc == "metadata":
            return self.hypercube[:, :, metadata.index.tolist()]
        else:
            return self.hypercube_aux[:, :, metadata.index.tolist()]

    def get_term(self, term) -> Tuple[pd.DataFrame, np.ndarray]:
        df_ref = self.metadata[self.metadata.term == term]
        cube_ref = self.hypercube[:, :, df_ref.index.tolist()]
        return df_ref, cube_ref
