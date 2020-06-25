import numpy as np
from scipy.stats import zscore
from sklearn.decomposition import PCA

from spikeloader import SpikeStimLoader


class NoSpontLoader(SpikeStimLoader):
    def __init__(self, *args, subtract_spont: bool = True, n_spont_comp: int = 25, **kwargs):
        super().__init__(*args, **kwargs)
        self.S_corr = self._subtract_spont(self.S, n_spont_comp) if subtract_spont else None
        self.n_spont_comp = n_spont_comp if subtract_spont else 0

    def _subtract_spont(self, S: np.ndarray, n_components: int) -> np.ndarray:
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
            np.where(np.isin(np.arange(np.max(self.frame_start) + 1), self.frame_start,
                             assume_unique=True, invert=True))[0]  # Invert indices.
        assert idx_spont.size + self.frame_start.size == self.spks.shape[0]

        S_spont = zscore(self.spks[idx_spont, :], axis=0)  # time x neu

        pca_spont = PCA(n_components=n_components).fit(S_spont)  # time x neu ; pcs are 'superneurons'
        pcs_spont = pca_spont.components_  # n_components x neu

        proj_spont = S @ pcs_spont.T  # neu x n_components
        S_corr = zscore(S - proj_spont @ pcs_spont, axis=0)

        return S_corr


if __name__ == '__main__':
    pl = NoSpontLoader()
