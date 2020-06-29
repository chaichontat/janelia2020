from __future__ import annotations

from functools import lru_cache, wraps

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from ..analyzer import Analyzer
from ..spikeloader import SpikeLoader


class ReceptiveField(Analyzer):
    def __init__(self, img_dim, λ: float = 1., seed=841):
        self.img_dim = img_dim
        self.λ = λ
        self.coef_ = np.empty(0)
        self.fit_type_ = 'None'
        self.seed = seed

    def _rf_decorator(func):
        """
        Generate receptive fields (RFs) by multivariate linear regression between spikes and pixels.
        Can generate RFs of each principal component (PC) OR each neuron.

        Parameters
        ----------
        S: np.ndarray
            Matrix of spikes for each stimulus (n_neu x n_stim).
        img: np.ndarray
            Matrix of stimulus images (y x x x n_stim)
        λ: float
            Coefficient for L2-regularization of the least-squares regression.
        n_pcs: int
            Number of PCs to generate RFs of. OR
            0 -> generate RF of each neuron.

        Returns
        -------
        B0: np.ndarray
            RFs in (y x x x `pca`) OR (y x x x n_stim).

        """

        @wraps(func)
        def rf_routines(self: ReceptiveField, imgs: np.ndarray = None, S: np.ndarray = None, *args, **kwargs):
            Sp = func(self, imgs, S, *args, **kwargs)

            # Linear regression
            print(f'Running linear regression with ridge coefficient {self.λ: .2f}.')
            ridge = Ridge(alpha=self.λ, random_state=np.random.RandomState(self.seed)).fit(imgs, Sp)
            self.coef_ = ridge.coef_
            return self

        return rf_routines

    def fit(self, *args, **kwargs):
        return self.fit_neuron(*args, **kwargs)

    @_rf_decorator
    def fit_neuron(self, imgs, S) -> ReceptiveField:
        self.fit_type_ = 'neu'
        return S

    @_rf_decorator
    def fit_pc(self, imgs, S, n_pc=30) -> ReceptiveField:
        pca_model = PCA(n_components=n_pc, random_state=np.random.RandomState(self.seed)).fit(S.T)
        self.fit_type_ = 'pcs'
        return pca_model.components_.T

    def transform(self, imgs=None):
        return imgs @ self.coef_.T

    def fit_transform(self, imgs=None, S=None):
        self.fit_neuron(imgs, S)
        return self.transform(imgs)

    def _reshape_rf(self, coef, smooth=0.5):
        B0 = np.reshape(coef, [self.img_dim[0], self.img_dim[1], -1])
        if smooth > 0:
            B0 = gaussian_filter(B0, [smooth, smooth, 0])
        return np.transpose(B0, (2, 0, 1))

    @property
    def rf_(self):
        return self._reshape_rf(self.coef_.T)

    @lru_cache
    def gen_rf_rank(self, n_pc):
        """ This `n_pc` is NOT the same as the `n_pc` in `fit_pc`. """
        adjusted_npca = min(3 * n_pc, *self.coef_.T.shape)  # Increase accuracy for randomized SVD.
        if adjusted_npca < n_pc:
            raise ValueError('Size of B lower than requested number of PCs.')

        model = PCA(n_components=adjusted_npca, random_state=np.random.RandomState(self.seed)).fit(self.coef_.T)

        B_reduced = self.coef_.T @ model.components_[:n_pc, :].T @ model.components_[:n_pc, :]
        return self._reshape_rf(B_reduced)

    def plot_rf(self, B0=None, figsize=(10, 8), nrows=5, ncols=4, dpi=300, title=None, save=None) -> None:
        """
        Generate a grid of RFs.

        Parameters
        ----------
        B0: np.ndarray
            RFs in (n_stim x y x x ).

        Returns
        -------
        None
        """
        if B0 is None:
            B0 = self.rf_
        assert B0.shape[0] >= nrows * ncols

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi, constrained_layout=True)
        axs = axs.flatten()
        for i in range(nrows * ncols):
            rfmax = np.max(np.abs(B0[:nrows * ncols, :, :]))
            axs[i].imshow(B0[i, :, :], cmap='bwr', vmin=-rfmax, vmax=rfmax)
            axs[i].axis('off')
            axs[i].set_title(f'PC {i + 1}')
        if title is not None:
            fig.suptitle(title)
        if save is not None:
            plt.savefig(save)
        plt.show()


if __name__ == '__main__':
    sns.set()
    loader = SpikeLoader.from_hdf5('data/processed.hdf5')

    # %% Generate RFs for PCs.
    rf = ReceptiveField(loader.img_dim)
    B = rf.fit_pc(loader.imgs_stim, loader.S, n_pc=30)
    rf.plot_rf()

    # # %% Generate RFs for every neuron.
    # rf = ReceptiveField(loader.img_dim)
    # rf.fit_neuron(loader.imgs_stim, loader.S)
    #
    # with open('field.pk', 'wb') as f:
    #     pickle.dump(rf, f)
