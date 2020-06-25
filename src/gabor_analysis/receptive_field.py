# %%
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from skopt import gp_minimize
from skopt.space import Real

from ..analyzer import Analyzer
from ..spikeloader import SpikeStimLoader


class ReceptiveField(Analyzer):
    def __init__(self, loader: SpikeStimLoader, n_pcs: int = 100, λ: float = 1.):
        self.loader = loader
        self.img_dim = self.loader.img.shape
        self.n_pcs = n_pcs
        self.λ = λ
        self.coef_ = np.empty(0)

    def fit(self, X=None, S=None):
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
        if S is None:
            S = self.loader.S
        if X is None:
            X = self.loader.X

        if self.n_pcs:
            pca_model = PCA(n_components=self.n_pcs).fit(S.T)
            Sp = pca_model.components_.T
        else:
            Sp = S

        # Linear regression
        print(f'Running linear regression with ridge coefficient {self.λ: .2f}.')
        ridge = Ridge(alpha=self.λ).fit(X, Sp)
        self.coef_ = ridge.coef_  # n_pcs x n_pxs
        return self

    def transform(self, X=None):
        if X is None:
            X = self.loader.X
        return X @ self.coef_.T

    def fit_transform(self, X=None, S=None):
        self.fit(X, S)
        return self.transform(X)

    @property
    def rf_(self):
        B0 = np.reshape(self.coef_.T, [self.img_dim[1], self.img_dim[2], -1])
        return np.transpose(gaussian_filter(B0, [.5, .5, 0]), (2, 0, 1))

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
        rfmax = np.max(np.abs(B0))
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi, constrained_layout=True)
        axs = axs.flatten()
        for i in range(nrows * ncols):
            axs[i].imshow(B0[i, :, :], cmap='bwr', vmin=-rfmax, vmax=rfmax)
            axs[i].axis('off')
            axs[i].set_title(f'PC {i + 1}')
        if title is not None:
            fig.suptitle(title)
        if save is not None:
            plt.savefig(save)
        plt.show()


def reduce_rf_rank(B: np.ndarray, n_pcs: int):
    assert n_pcs > 0
    B_flatten = np.reshape(B, (len(B), -1))  # n_neu x pixels.
    adjusted_npca = min(3 * n_pcs, *B_flatten.shape)  # Increase accuracy for randomized SVD.
    if adjusted_npca < n_pcs:
        raise ValueError('Size of B lower than requested number of PCs.')

    model = PCA(n_components=adjusted_npca)
    B_reduced = (model.fit_transform(B_flatten) @ model.components_).reshape(B.shape)
    pcs = model.components_.reshape((adjusted_npca, B.shape[1], B.shape[2]))
    return B_reduced[:, :n_pcs], pcs[:n_pcs, ...]


if __name__ == '__main__':
    sns.set()
    loader = SpikeStimLoader()

    # %% Generate RFs for PCs.
    rf = ReceptiveField(loader, n_pcs=50)
    B = rf.fit()
    rf.plot_rf(B)

    # %% Generate RFs for every neuron.
    rf = ReceptiveField(loader, n_pcs=0, λ=1.1)
    rf.fit()

    with open('gabor_analysis/field.pk', 'wb') as f:
        pickle.dump(rf, f)

    # %% CV
    trX, teX, trS, teS = loader.train_test_split()
    def objective(λ):
        print(f'{λ=}')
        λ = λ[0]
        model = ReceptiveField(loader, n_pcs=0, λ=λ).fit(trX, trS)
        S_hat = model.transform(teX)
        mse = mean_squared_error(teS, S_hat)

        model = ReceptiveField(loader, n_pcs=0, λ=λ).fit(teX, teS)
        S_hat = model.transform(trX)
        mse += mean_squared_error(trS, S_hat)
        return float(mse)

    space = [
        Real(0.1, 100, prior='log-uniform', name='λ')
    ]

    res_gp = gp_minimize(objective, space, n_calls=20, n_random_starts=10, random_state=439,
                         verbose=True)

    # %% Save two sets of RFs
    trX, teX, trS, teS = loader.train_test_split()
    rf.fit(trX, trS)
    with open(f'gabor_analysis/field1.pk', 'wb') as f:
        pickle.dump(rf, f)

    rf = ReceptiveField(loader, n_pcs=0, λ=1.1)
    rf.fit(teX, teS)
    with open(f'gabor_analysis/field2.pk', 'wb') as f:
        pickle.dump(rf, f)
