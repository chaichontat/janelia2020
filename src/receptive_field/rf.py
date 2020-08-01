from __future__ import annotations

from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import Ridge

from ..analyzer import Analyzer
from ..spikeloader import SpikeLoader
from ..utils.io import hdf5_save


class ReceptiveField(Analyzer):
    HYPERPARAMS = ['img_dim', 'lamda', 'smooth', 'seed', 'fit_type_']
    ARRAYS = ['coef_', 'transformed']
    DATAFRAMES = None

    def __init__(self, img_dim, lamda: float = 1., smooth=0.5, seed=841, **kwargs):
        super().__init__(**kwargs)
        self.img_dim = img_dim
        self.lamda = lamda
        self.seed = seed
        self.smooth = smooth

        self.pca_model = None

    def _rf_routines(self: ReceptiveField, imgs: np.ndarray = None, S: np.ndarray = None, *args, **kwargs):
        ridge = Ridge(alpha=self.lamda, random_state=np.random.RandomState(self.seed)).fit(imgs, S)
        self.coef_ = ridge.coef_
        return self

    def fit(self, *args, **kwargs):
        print('Fitting neuron by default. To fit PC, run `fit_pc`.')
        return self.fit_neuron(*args, **kwargs)

    def fit_neuron(self, imgs, S) -> ReceptiveField:
        self.fit_type_ = 'Neuron'
        return self._rf_routines(imgs, S)

    def fit_pc(self, imgs, S, n_pc=30) -> ReceptiveField:
        self.pca_model = PCA(n_components=n_pc, random_state=np.random.RandomState(self.seed)).fit(S.T)
        self.fit_type_ = 'PC'
        return self._rf_routines(imgs, self.pca_model.components_.T)

    def transform(self, imgs):
        self.transformed = imgs @ self.coef_.T
        return self.transformed

    def fit_transform(self, imgs, S):
        self.fit_neuron(imgs, S)
        return self.transform(imgs)

    @staticmethod
    def reshape_rf(coef, img_dim, smooth=0.5):
        B0 = np.reshape(coef, [img_dim[0], img_dim[1], -1])
        if smooth > 0:
            B0 = gaussian_filter(B0, [smooth, smooth, 0])
        return np.transpose(B0, (2, 0, 1))

    @property
    def rf_(self):
        return self.reshape_rf(self.coef_.T, self.img_dim, self.smooth)

    def plot(self, B0=None, random=False,
             figsize=(10, 8), nrows=5, ncols=4, dpi=300, title=None, save=None) -> None:
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

        n_fig = nrows * ncols
        if random:
            idx = sorted(np.random.RandomState(self.seed).randint(low=0, high=len(self.rf_), size=n_fig))
        else:
            idx = np.arange(n_fig)

        for i in range(nrows * ncols):
            rfmax = np.max(np.abs(B0[idx, :, :]))
            axs[i].imshow(B0[idx[i], :, :], cmap="twilight_shifted", vmin=-rfmax, vmax=rfmax)
            axs[i].axis('off')
            axs[i].set_title(f'{self.fit_type_} {idx[i]}')
        if title is not None:
            fig.suptitle(title)
        if save is not None:
            plt.savefig(save)
        plt.show()


class ReducedRankReceptiveField(ReceptiveField):
    HYPERPARAMS = ReceptiveField.HYPERPARAMS + ['rank']
    ARRAYS = ReceptiveField.ARRAYS + ['coef_', 'coef_full_rank']

    def __init__(self, *args, rank=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = rank

    def _rf_routines(self, imgs: np.ndarray = None, S: np.ndarray = None, *args, **kwargs):
        ridge = Ridge(alpha=self.lamda, random_state=np.random.RandomState(self.seed)).fit(imgs, S)
        self.coef_full_rank = ridge.coef_
        self.coef_ = self._reduce_rank(ridge.coef_)
        return self

    def _reduce_rank(self, coef):
        mean = np.mean(coef, axis=0)
        coef -= mean
        svd: TruncatedSVD = TruncatedSVD(n_components=self.rank, random_state=self.seed).fit(coef)
        return svd.inverse_transform(svd.transform(coef)) + mean


def gen_rf_rank(rfs, n_pc, seed=455):
    """ This `n_pc` is NOT the same as the `n_pc` in `fit_pc`. """
    rfs_shape = rfs.shape
    coef = rfs.reshape([rfs.shape[0], -1])

    adjusted_npca = min(3 * n_pc, *coef.shape)  # Increase accuracy for randomized SVD.
    if adjusted_npca < n_pc:
        raise ValueError('Size of B lower than requested number of PCs.')

    model = PCA(n_components=adjusted_npca, random_state=np.random.RandomState(seed)).fit(coef)
    X = model.components_[:n_pc, :]
    B_reduced = coef @ X.T @ X
    return ReceptiveField.reshape_rf(B_reduced.T, rfs_shape[1:])


def make_regression_truth():
    loader = SpikeLoader.from_hdf5('tests/data/processed.hdf5')
    rf = ReceptiveField(loader.img_dim)
    rf.fit_neuron(loader.imgs_stim, loader.S)
    neu = rf.rf_
    rf.fit_pc(loader.imgs_stim, loader.S)
    pc = rf.rf_
    hdf5_save('tests/data/regression_test_data.hdf5', 'ReceptiveField', arrs={'neu': neu, 'pc': pc},
              append=True, overwrite_group=True)


if __name__ == '__main__':
    sns.set()
    loader = SpikeLoader.from_hdf5('data/processed.hdf5')

    # %% Generate RFs for PCs.
    rf = ReceptiveField(loader.img_dim)
    rf.fit_pc(loader.imgs_stim, loader.S, n_pc=30)
    rf.plot()

    # %% Generate RFs for every neuron.
    rf = ReceptiveField(loader.img_dim)
    rf.fit_neuron(loader.imgs_stim, loader.S)
    rf.plot(random=True)
