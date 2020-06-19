from typing import Union

import jax.numpy as np
import numpy as onp
from jax.numpy.linalg import inv, cholesky
from sklearn.decomposition import PCA

from spikeloader import SpikeLoader

Arrays = Union[np.DeviceArray, onp.ndarray]


class CanonicalRidge:
    """
    Class for canonical correlation analysis with ridge regularization.
    See http://www2.imm.dtu.dk/pubdb/edoc/imm4981.pdf

    Uses JAX for faster numerical computation.
    Based on the scikit-learn Estimator API.

    """

    def __init__(self, n: int = 25, lx: float = 0.95, ly: float = 0.95):
        self.n = n
        self.λx = lx
        self.λy = ly
        self.V, self.singular_values, self.Σ, self.U = 4 * [None]

    def fit(self, X: Arrays, Y: Arrays):
        K = self._calc_canon_mat(X, Y)
        model = PCA(self.n).fit(K)  # Randomized SVD.

        # Based on X = UΣV^T.
        self.V = model.components_.T
        self.singular_values = self.Σ = np.diag(model.singular_values_)
        self.U = K @ self.V @ np.linalg.inv(self.Σ)
        return self

    def fit_transform(self, X: Arrays, Y: Arrays) -> np.DeviceArray:
        self.fit(X, Y)
        return self.transform(X, Y)

    def transform(self, X: Arrays, Y: Arrays) -> np.DeviceArray:
        return X @ self.U, Y @ self.V

    def calc_canon_coef(self, X: Arrays, Y: Arrays) -> np.DeviceArray:
        X_p, Y_p = self.transform(X, Y)
        return np.array([onp.corrcoef(X_p[:, i], Y_p[:, i])[0, 1] for i in range(self.n)])

    def _calc_canon_mat(self, X: Arrays, Y: Arrays) -> np.DeviceArray:
        X, Y = np.asarray(X), np.asarray(Y)

        X -= np.mean(X, axis=0)
        Y -= np.mean(Y, axis=0)
        K = inv(cholesky((1 - self.λx) * np.cov(X, rowvar=False) + self.λx * np.eye(X.shape[1]))) @ \
            (X.T @ Y) / (X.T.shape[0] * Y.shape[1]) @ \
            inv(cholesky((1 - self.λy) * np.cov(Y, rowvar=False) + self.λy * np.eye(Y.shape[1])))

        return K


def prepare_data(path='superstim.npz'):
    data = SpikeLoader(path)
    split = onp.random.rand(data.S_corr.shape[1]) > 0.5
    inv_split = np.in1d(np.arange(data.S_corr.shape[1]), split, assume_unique=True, invert=True)

    train = [data.S_corr[data.ypos >= 210][:, split].T, data.S_corr[data.ypos < 210][:, split].T]
    test = [data.S_corr[data.ypos >= 210][:, inv_split].T, data.S_corr[data.ypos < 210][:, inv_split].T]
    return train, test


if __name__ == '__main__':
    train, test = prepare_data()
    X, Y = np.asarray(train[0]), np.asarray(train[1])

    # %%
    print('Running CCA.')
    model = CanonicalRidge().fit(X, Y)
    tr = model.transform(*train)
    te = model.transform(*test)

    # %%
    # fig, axs = plt.subplots(nrows=2, ncols=2, dpi=300, figsize=(10, 8), constrained_layout=True)
    #
    # axs[0, 0].scatter(tr1[:, 0], tr2[:, 0], s=2, alpha=0.5, linewidth=0)
    # axs[0, 0].set_title(f'Train: first CC, Corr: {onp.corrcoef(tr1[:, 0], tr2[:, 0])[0, 1]:.3f}')
    # axs[0, 1].scatter(tr1[:, 1], tr2[:, 1], s=2, alpha=0.5, linewidth=0)
    # axs[0, 1].set_title(f'Train: second CC, Corr: {onp.corrcoef(tr1[:, 1], tr2[:, 1])[0, 1]:.3f}')
    # axs[1, 0].scatter(te1[:, 0], te2[:, 0], s=2, alpha=0.5, linewidth=0)
    # axs[1, 0].set_title(f'Test: second CC, Corr: {onp.corrcoef(te1[:, 0], te2[:, 0])[0, 1]:.3f}')
    # axs[1, 1].scatter(te1[:, 1], te2[:, 1], s=2, alpha=0.5, linewidth=0)
    # axs[1, 1].set_title(f'Test: second CC, Corr: {onp.corrcoef(te1[:, 1], te2[:, 1])[0, 1]:.3f}')
    #
    # axs = axs.flatten()
    # for ax in axs:
    #     ax.set_xlabel('V1')
    #     ax.set_ylabel('V2')
    #
    # fig.suptitle('CCA w/ diagonal cov matrix. Train/test split in half.')
    # plt.show()
