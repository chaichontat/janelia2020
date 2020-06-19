import numpy as np
from scipy.stats import zscore
from sklearn.decomposition import PCA


class SpikeLoader:
    """
    Class to load suite2p npz files with stimulus information.
    Can subtract spontaneous PCs.
    """

    def __init__(self, path: str = '../superstim.npz',
                 subtract_spont: bool = True, n_spont_comp: int = 25):
        print('Loading data.')
        with np.load(path) as npz:
            self.img, self.spks = npz['img'], npz['spks']
            self.xpos, self.ypos = npz['xpos'], npz['ypos']
            self.frame_start, self.istim = npz['frame_start'], npz['istim']

        assert len(self.frame_start) == len(self.istim)
        self.S = zscore(self.spks[:, self.frame_start], axis=1)  # neu x time
        self.S_corr = self._subtract_spont(self.S, n_spont_comp) if subtract_spont else None
        self.n_spont_comp = n_spont_comp if subtract_spont else 0

    def _subtract_spont(self, S: np.ndarray, n_components: int) -> np.ndarray:
        """
        Project S onto the spontaneous activities subspace and subtract out.

        Parameters
        ----------
        S: np.ndarray
            Non-spontaneous spiking data (neu x stim).
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
        assert idx_spont.size + self.frame_start.size == self.spks.shape[1]

        S_spont = zscore(self.spks[:, idx_spont], axis=1)  # neu x time

        pca_spont = PCA(n_components=n_components).fit(S_spont.T)  # time x neu ; pcs are 'superneurons'
        pcs_spont = pca_spont.components_  # n_components x neu

        proj_spont = S.T @ pcs_spont.T  # neu x n_components
        S_corr = zscore((S.T - proj_spont @ pcs_spont).T, axis=1)  # neu x time

        return S_corr
