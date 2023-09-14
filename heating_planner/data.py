from typing import List, Tuple
import numpy as np
import pandas as pd
from heating_planner.utils import load_np_from_anywhere, load_pil_from_anywhere


def metadata2featurename(variable, season):
    return " _".join([variable, season])


def serie2featurename(meta: pd.Series):
    return metadata2featurename(meta.variable, meta.season)


class Datum:
    def __init__(
        self,
        metadata_path,
        metadata_aux_path,
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

    def get_every_features_names(self):
        feats = self.metadata.apply(serie2featurename, axis=1).tolist()
        feats_aux = self.metadata.apply(serie2featurename, axis=1).tolist()
        return list(set(feats_aux + feats))

    def get_every_seasons(self, with_variable_in: List | str | None = None):
        df = pd.concat([self.metadata, self.metadata_aux])
        if with_variable_in:
            if isinstance(with_variable_in, str):
                with_variable_in = [with_variable_in]
            with_variable_in = set(with_variable_in)
            df = df[df.variable.isin(with_variable_in)]
        return df.season.unique().tolist()

    def get_every_variables(self, with_season_in: List | str | None = None):
        df = pd.concat([self.metadata, self.metadata_aux])
        if with_season_in:
            if isinstance(with_season_in, str):
                with_season_in = [with_season_in]
            with_season_in = set(with_season_in)
            df = df[df.season.isin(with_season_in)]
        return df.variable.unique().tolist()
