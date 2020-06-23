#%%
import pickle
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.ndimage import gaussian_filter
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from typing import Tuple

from sklearn.model_selection import train_test_split

from spikeloader import SpikeLoader


class ReceptiveField(SpikeLoader):
    def __init__(self, *args, img_scale: float = 0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_scale = img_scale

        with np.load(self.path) as npz:
            self.img = np.transpose(npz['img'], (2, 0, 1))  # stim x y x x

        self.img = ndi.zoom(self.img, (1, self.img_scale, self.img_scale), order=1)

        # Normalized.
        self.X = np.reshape(self.img[self.istim, ...], [len(self.istim), -1])
        self.X = zscore(self.X, axis=0) / np.sqrt(len(self.istim))  # (stim x pxs)

    def train_test_split(self, test_size: float = 0.5, random_state: int = 1256) -> Tuple:
        return train_test_split(self.X, self.S, test_size=test_size, random_state=random_state)

    def gen_rf(self, X=None, S=None, n_pcs: int = 100, λ: float = 1.) -> np.ndarray:
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
            S = self.S
        if X is None:
            X = self.X

        if n_pcs:
            print('PCAing.')
            pca_model = PCA(n_components=n_pcs).fit(S.T)
            Sp = pca_model.components_.T
        else:
            Sp = S

        # Linear regression
        print(f'Running linear regression with ridge coefficient {λ: .2f}.')
        # ridge = Ridge(alpha=λ).fit(X, Sp)
        B0 = np.linalg.solve((X.T @ X + λ * np.eye(X.shape[1])), X.T @ Sp)
        B0 = np.reshape(B0, [self.img.shape[1], self.img.shape[2], -1])
        B0 = np.transpose(gaussian_filter(B0, [.5, .5, 0]), (2, 0, 1))
        return B0

    @staticmethod
    def plot_rf(B0: np.ndarray, figsize=(10, 8), nrows=5, ncols=4, dpi=300, title=None, save=None) -> None:
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


if __name__ == '__main__':
    sns.set()

    #%% Generate RFs for PCs.
    rf = ReceptiveField(subtract_spont=False)
    B = rf.gen_rf(n_pcs=50)
    rf.plot_rf(B)

    # #%% Generate RFs for every neuron.
    # B = rf.gen_rf(n_pcs=0)
    # with open('gabor_analysis/field.pk', 'wb') as f:
    #     pickle.dump(B, f)

    # #%% Split data randomly into two and fit RFs on each.
    # trX, teX, trS, teS = rf.train_test_split()
    #
    # B = rf.gen_rf(trX, trS, n_pcs=0)
    # with open(f'gabor_analysis/field1.pk', 'wb') as f:
    #     pickle.dump(B, f)
    #
    # B = rf.gen_rf(teX, teS, n_pcs=0)
    # with open(f'gabor_analysis/field2.pk', 'wb') as f:
    #     pickle.dump(B, f)
