import numpy as np
from scipy.stats import zscore
from sklearn.decomposition import PCA

from src.analyzer import Analyzer


class SubtractSpontAnalyzer(Analyzer):

    params = ['n_pc', 'seed']
    arrs = ['pcs']
    dfs = None

    def __init__(self, n_pc: int = 25, seed: int = 437,
                 pcs: np.ndarray = None):
        super().__init__()
        self.n_pc = n_pc
        self.pcs = pcs
        self.seed = seed
        assert self.n_pc > 0

    def fit(self, spks, idx_spont):
        sponts = zscore(spks[idx_spont, :], axis=0)

        n_used = min(3 * self.n_pc, *sponts.shape)  # Randomized SVD.
        pca_spont = PCA(n_components=n_used, random_state=np.random.RandomState(self.seed)).fit(sponts)  # time x neu
        self.pcs = pca_spont.components_.T[:, :self.n_pc]  # neu x n_components
        return self

    def transform(self, S):
        proj_spont = S @ self.pcs  # neu x n_components
        return zscore(S - proj_spont @ self.pcs.T, axis=0)
