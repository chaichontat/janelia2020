import numpy as np
from scipy.stats import zscore
from sklearn.decomposition import PCA

from spikeloader import SpikeLoader


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
        self.n_spont_pc = n_spont_pc
        if self.n_spont_pc > 0:
            print(f'Subtracting {n_spont_pc} spontaneous components.')
            self.S_nospont = self.subtract_spont(self.loader.S)
        else:
            print(f'No spontaneous subtraction. `self.S_nospont` not loaded.')
            self.S_nospont = self.loader.S

    def subtract_spont(self, S: np.ndarray) -> np.ndarray:
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
        idx_spont = \
            np.where(np.isin(np.arange(np.max(self.loader.frame_start) + 1), self.loader.frame_start,
                             assume_unique=True, invert=True))[0]  # Invert indices.
        assert idx_spont.size + self.loader.frame_start.size == self.loader.spks.shape[0]

        S_spont = zscore(self.loader.spks[idx_spont, :], axis=0)  # time x neu

        n_used = min(3 * self.n_spont_pc, *S_spont.shape)  # Randomized SVD.
        pca_spont = PCA(n_components=n_used).fit(S_spont)  # time x neu
        pcs_spont = pca_spont.components_.T[:, :self.n_spont_pc]  # neu x n_components

        proj_spont = S @ pcs_spont  # neu x n_components
        self.S_nospont = zscore(S - proj_spont @ pcs_spont.T, axis=0)

        return self.S_nospont
