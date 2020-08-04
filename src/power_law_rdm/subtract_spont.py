from __future__ import annotations

import numpy as np
from scipy.stats import zscore
from sklearn.decomposition import PCA

from src.analyzer import Analyzer


class SubtractSpontAnalyzer(Analyzer):

    HYPERPARAMS = ["n_pc", "seed"]
    ARRAYS = ["pcs"]
    DATAFRAMES = None

    def __init__(self, n_pc: int = 25, seed: int = 437, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_pc = n_pc
        self.seed = seed
        self.pca: PCA
        assert self.n_pc > 0

    def fit(self, spks: np.ndarray, idx_spont: np.ndarray) -> SubtractSpontAnalyzer:
        sponts = spks[idx_spont, :]
        
        self.pca = PCA(n_components=self.n_pc, random_state=np.random.RandomState(self.seed)).fit(
            sponts
        )  # time x neu
        self.pcs = self.pca.components_.T  # neu x n_components
        return self

    def transform(self, S: np.ndarray) -> np.ndarray:
        proj_spont = S @ self.pcs  # neu x n_components
        return S - proj_spont @ self.pcs.T
