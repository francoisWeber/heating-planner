import pandas as pd
import numpy as np
from abc import abstractmethod


class DriasScoring:
    @abstractmethod
    def apply(self): ...


class WeightedDriasScoring(DriasScoring):
    def __init__(self, df: pd.DataFrame, coefs: dict):
        self.df = df
        self.coefs = coefs

    def apply(self):
        data = self.df[list(self.coefs.keys())].to_numpy()
        ordered_coefs = np.array(list(self.coefs.values())).reshape(-1, 1)
        return data.dot(ordered_coefs)
