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
        assert self.n_pc > 0

    def fit(self, spks: np.ndarray, idx_spont: np.ndarray) -> SubtractSpontAnalyzer:
        sponts = zscore(spks[idx_spont, :], axis=0)

        n_used = min(2 * self.n_pc, *sponts.shape)  # Randomized SVD.
        pca_spont = PCA(n_components=n_used, random_state=np.random.RandomState(self.seed)).fit(
            sponts
        )  # time x neu
        self.pcs = pca_spont.components_.T[:, : self.n_pc]  # neu x n_components
        return self

    def transform(self, S: np.ndarray) -> np.ndarray:
        mean = np.mean(S, axis=0)
        S -= mean
        proj_spont = S @ self.pcs  # neu x n_components
        return zscore(S - proj_spont @ self.pcs.T, axis=0) + mean
