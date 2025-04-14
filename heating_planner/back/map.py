import pandas as pd
from heating_planner.back.data import DriasDataset
from heating_planner.back.geo import project_to_grid
from heating_planner.back.scoring import DriasScoring

import numpy as np

GRID_SIZE = 120


class DriasMap:
    def __init__(self, dataset: DriasDataset, grid_size=GRID_SIZE):
        self.grid_size = grid_size
        self.df = self.prepare_mapped_dataset(dataset.df)

    def prepare_mapped_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        x, y = project_to_grid(df.latitude.to_numpy(), df.longitude.to_numpy(), grid_size=self.grid_size)
        df["x"] = x
        df["y"] = (self.grid_size - 1) - y  # because of img inversion
        return df

    def prepare_score_map(self, scoring: DriasScoring):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        scores = scoring.apply()
        grid[self.df.y, self.df.x] = scores.ravel()

        return np.expand_dims(grid, axis=2)
