from __future__ import annotations
import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import Ridge

from ..analyzer import Analyzer
from ..spikeloader import SpikeLoader
from ..utils.io import hdf5_save


class ReceptiveField(Analyzer):
    HYPERPARAMS = ["img_dim", "lamda", "smooth", "seed", "fit_type_"]
    ARRAYS = ["coef_", "transformed"]
    DATAFRAMES = None

    """Ridge regression of spiking data to images.
    
    Can regress on
        - SpikeLoader.S
        - PCs of SpikeLoader.S
        
    Args:
        img_dim (np.ndarray): img_dim from SpikeLoader.
        lambda (float): Ridge coefficient.
        smooth (float): Width of smoothing Gaussian kernel for final RF. 0 for no smoothing.
        seed (int): Random seed.
        
    Attributes: All of args and
        fit_type_ (str): "PC" or "Neuron" for recordkeeping purpose.
        coef_ (np.ndarray): Regression matrix of flattened image.
        pca_model (PCA): PCA instance from sklearn if fit_pc.
    
    """

    def __init__(self, img_dim, lamda: float = 1.0, smooth=0.5, seed=841, **kwargs) -> None:
        super().__init__(**kwargs)
        self.img_dim = img_dim
        self.lamda = lamda
        self.seed = seed
        self.smooth = smooth

        self.pca_model = None

    def _rf_routines(self, imgs: np.ndarray, S: np.ndarray) -> ReceptiveField:
        """ Common operations to both {fit_neuron} and {fit_pc}. """
        ridge = Ridge(alpha=self.lamda, random_state=np.random.RandomState(self.seed)).fit(
            imgs, S
        )
        self.coef_ = ridge.coef_
        return self

    def fit(self, *args, **kwargs) -> ReceptiveField:
        logging.info("Fitting neuron by default. To fit PC, run `fit_pc`.")
        return self.fit_neuron(*args, **kwargs)

    def fit_neuron(self, imgs: np.ndarray, S: np.ndarray) -> ReceptiveField:
        self.fit_type_ = "Neuron"
        return self._rf_routines(imgs, S)

    def fit_pc(self, imgs: np.ndarray, S: np.ndarray, n_pc: int = 30) -> ReceptiveField:
        self.pca_model = PCA(
            n_components=n_pc, random_state=np.random.RandomState(self.seed)
        ).fit(S.T)
        self.fit_type_ = "PC"
        return self._rf_routines(imgs, self.pca_model.components_.T)

    def transform(self, imgs: np.ndarray) -> np.ndarray:
        self.transformed = imgs @ self.coef_.T
        return self.transformed

    def fit_transform(self, imgs: np.ndarray, S: np.ndarray) -> np.ndarray:
        self.fit_neuron(imgs, S)
        return self.transform(imgs)

    @staticmethod
    def reshape_rf(coef: np.ndarray, img_dim: np.ndarray, smooth: float = 0.5) -> np.ndarray:
        B0 = np.reshape(coef, [img_dim[0], img_dim[1], -1])
        if smooth > 0:
            B0 = gaussian_filter(B0, [smooth, smooth, 0])
        return np.transpose(B0, (2, 0, 1))

    @property
    def rf_(self) -> np.ndarray:
        return self.reshape_rf(self.coef_.T, self.img_dim, self.smooth)

    def plot(
        self,
        B0=None,
        random=False,
        figsize=(10, 8),
        nrows=5,
        ncols=4,
        dpi=300,
        title=None,
        save=None,
    ) -> None:
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

        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi, constrained_layout=True
        )
        axs = axs.flatten()

        n_fig = nrows * ncols
        if random:
            idx = sorted(
                np.random.RandomState(self.seed).randint(low=0, high=len(self.rf_), size=n_fig)
            )
        else:
            idx = np.arange(n_fig)

        for i in range(nrows * ncols):
            rfmax = np.max(np.abs(B0[idx, :, :]))
            axs[i].imshow(B0[idx[i], :, :], cmap="twilight_shifted", vmin=-rfmax, vmax=rfmax)
            axs[i].axis("off")
            axs[i].set_title(f"{self.fit_type_} {idx[i]}")
        if title is not None:
            fig.suptitle(title)
        if save is not None:
            plt.savefig(save)
        plt.show()


class ReducedRankReceptiveField(ReceptiveField):
    HYPERPARAMS = ReceptiveField.HYPERPARAMS + ["rank"]
    ARRAYS = ReceptiveField.ARRAYS + ["coef_", "coef_full_rank"]

    def __init__(self, *args, rank: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = rank

    def _rf_routines(self, imgs: np.ndarray, S: np.ndarray) -> ReducedRankReceptiveField:
        ridge = Ridge(alpha=self.lamda, random_state=np.random.RandomState(self.seed)).fit(
            imgs, S
        )
        self.coef_full_rank = ridge.coef_
        self.coef_ = self._reduce_rank(ridge.coef_)
        return self

    def _reduce_rank(self, coef: np.ndarray) -> np.ndarray:
        mean = np.mean(coef, axis=0)
        coef -= mean
        svd: TruncatedSVD = TruncatedSVD(n_components=self.rank, random_state=self.seed).fit(
            coef
        )
        return svd.inverse_transform(svd.transform(coef)) + mean


def gen_rf_rank(rfs: np.ndarray, n_pc: int, seed: int = 455) -> np.ndarray:
    """ This `n_pc` is NOT the same as the `n_pc` in `fit_pc`. """
    rfs_shape = rfs.shape
    coef = rfs.reshape([rfs.shape[0], -1])

    adjusted_npca = min(3 * n_pc, *coef.shape)  # Increase accuracy for randomized SVD.
    if adjusted_npca < n_pc:
        raise ValueError("Size of B lower than requested number of PCs.")

    model = PCA(n_components=adjusted_npca, random_state=np.random.RandomState(seed)).fit(coef)
    X = model.components_[:n_pc, :]
    B_reduced = coef @ X.T @ X
    return ReceptiveField.reshape_rf(B_reduced.T, rfs_shape[1:])


def gen_test_data(path: str) -> None:
    loader = SpikeLoader.from_hdf5(path)
    rf = ReceptiveField(loader.img_dim)
    rf.fit_neuron(loader.imgs_stim, loader.S)
    neu = rf.rf_
    rf.fit_pc(loader.imgs_stim, loader.S)
    pc = rf.rf_
    hdf5_save(
        path, "ReceptiveField", arrs={"neu": neu, "pc": pc}, append=True, overwrite_group=True
    )


# if __name__ == '__main__':
#     gen_test_data("data/test.hdf5")
