import numpy as np
from scipy.stats import zscore
from sklearn.decomposition import PCA

from .spikeloader import SpikeLoader


class Analyzer:
    """
    Abstract class for data analysis from raw spike data in the form of `SpikeLoader` instance.
    Prevent `SpikeLoader` from being pickled along with the instance.

    All parameters required for a exact replication from an identical `SpikeLoader` instance
    and any data required for a proper functioning of helper functions should be pointed to by an instance variable.

    Any saved analysis should include proper context.

    """

    def __getstate__(self):
        d = self.__dict__.copy()
        try:
            del d['loader']
        except KeyError:
            pass
        return d


class SubtractSpontAnalyzer(Analyzer):
    def __init__(self, loader: SpikeLoader, n_spont_pc: int = 25):
        self.loader = loader
        self.loader_path = self.loader.path
        self.n_spont_pc = n_spont_pc
        self._S_nospont = np.empty(0)  # To allow pickling.

    @property
    def S_nospont(self):
        if len(self._S_nospont) == 0:
            if self.n_spont_pc > 0:
                print(f'Subtracting {self.n_spont_pc} spontaneous components.')
                S_spont = self.get_spont()
                self._S_nospont = self.subtract_spont(self.loader.S, S_spont).astype(np.float32)
            else:
                print(f'No spontaneous subtraction. `self.S_nospont` not loaded.')
                self._S_nospont = self.loader.S

        return self._S_nospont

    def get_spont(self) -> np.ndarray:
        idx_spont = \
            np.where(np.isin(np.arange(np.max(self.loader.istim.index) + 1), self.loader.istim.index,
                             assume_unique=True, invert=True))[0]  # Invert indices.
        assert idx_spont.size + self.loader.istim.index.size == self.loader.spks.shape[0]
        return zscore(self.loader.spks[idx_spont, :], axis=0)  # time x neu

    def subtract_spont(self, S: np.ndarray, S_spont: np.ndarray) -> np.ndarray:
        """
        Project S onto the spontaneous activities subspace and subtract out.

        Parameters
        ----------
        S: np.ndarray
            Non-spontaneous spiking data (stim x neu).
        n_components: int
            Number of spontaneous PCs to subtract.

        Returns
        -------
        S_corr: np.ndarray
            S after subtraction

        """

        n_used = min(3 * self.n_spont_pc, *S_spont.shape)  # Randomized SVD.
        pca_spont = PCA(n_components=n_used).fit(S_spont)  # time x neu
        pcs_spont = pca_spont.components_.T[:, :self.n_spont_pc]  # neu x n_components

        proj_spont = S @ pcs_spont  # neu x n_components
        return zscore(S - proj_spont @ pcs_spont.T, axis=0)
