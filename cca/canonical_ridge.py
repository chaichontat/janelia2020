from typing import Tuple, Union

import jax.numpy as np
import numpy as onp
from jax.numpy.linalg import inv, cholesky
from sklearn.decomposition import PCA

from analyzer import Analyzer
from spikeloader import SpikeLoader

Arrays = Union[np.DeviceArray, onp.ndarray]


class CanonicalRidge(Analyzer):
    """
    Class for canonical correlation analysis with ridge regularization.
    See http://www2.imm.dtu.dk/pubdb/edoc/imm4981.pdf

    Uses JAX for faster numerical computation.
    Based on the scikit-learn Estimator API.

    """

    def __init__(self, n: int = 25, lx: float = 0.85, ly: float = 0.85):
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

    def fit_transform(self, X: Arrays, Y: Arrays) -> Tuple[np.DeviceArray, np.DeviceArray]:
        self.fit(X, Y)
        return self.transform(X, Y)

    def transform(self, X: Arrays, Y: Arrays) -> Tuple[np.DeviceArray, np.DeviceArray]:
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

    def subtract_canon_comp(self, X: Arrays, Y: Arrays) -> Tuple[np.DeviceArray, np.DeviceArray]:
        X_comp, Y_comp = model.transform(X, Y)
        return X - (X_comp @ model.U.T), Y - (Y_comp @ model.V.T)


# %%
def prepare_train_test(loader: SpikeLoader):
    train, test = loader.train_test_split()

    # Split V1 and V2
    train = [train[:, loader.ypos >= 210], train[:, loader.ypos < 210]]
    test = [test[:, loader.ypos >= 210], test[:, loader.ypos < 210]]
    return train, test


# %%

if __name__ == '__main__':
    loader = SpikeLoader()
    train, test = prepare_train_test(loader)
    X, Y = np.asarray(train[0]), np.asarray(train[1])

    # %%
    print('Running CCA.')
    V1, V2 = loader.S[:, loader.ypos >= 210], loader.S[:, loader.ypos < 210]
    model = CanonicalRidge().fit(V1, V2)
    # te = model.transform(*test)

    # %% Subtract CCs
    V1_comp, V2_comp = model.transform(V1, V2)
    V1_nocc, V2_nocc = model.subtract_canon_comp(V1, V2)
    # %%
    import pickle

    with open('common_cc.pk', 'wb') as f:
        pickle.dump(model, f)

    # #%%
    # model1 = PCA(n_components=30).fit(V1_nocc)
    # model2 = PCA(n_components=30).fit(V2_nocc)
    #
    # #%%
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    #
    # fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(10, 6), dpi=300, constrained_layout=True)
    #
    # axs = np.hstack([axs[0:2, :], axs[2:4, :]]).T
    # for i in range(10):
    #     scale = np.max(np.abs([B[idx[i], :, :], B_reduced[idx[i], :, :]]))
    #     axs[i, 0].imshow(B[idx[i], :, :], cmap='bwr', vmin=-scale, vmax=scale)
    #     axs[i, 1].imshow(B_reduced[idx[i], :, :], cmap='bwr', vmin=-scale, vmax=scale)
    #     axs[i, 0].axis('off')
    #     axs[i, 1].axis('off')
    #
    #
    # # fig, axs = plt.subplots(nrows=2, ncols=2, dpi=300, figsize=(10, 8), constrained_layout=True)
    # #
    # # axs[0, 0].scatter(tr1[:, 0], tr2[:, 0], s=2, alpha=0.5, linewidth=0)
    # # axs[0, 0].set_title(f'Train: first CC, Corr: {onp.corrcoef(tr1[:, 0], tr2[:, 0])[0, 1]:.3f}')
    # # axs[0, 1].scatter(tr1[:, 1], tr2[:, 1], s=2, alpha=0.5, linewidth=0)
    # # axs[0, 1].set_title(f'Train: second CC, Corr: {onp.corrcoef(tr1[:, 1], tr2[:, 1])[0, 1]:.3f}')
    # # axs[1, 0].scatter(te1[:, 0], te2[:, 0], s=2, alpha=0.5, linewidth=0)
    # # axs[1, 0].set_title(f'Test: second CC, Corr: {onp.corrcoef(te1[:, 0], te2[:, 0])[0, 1]:.3f}')
    # # axs[1, 1].scatter(te1[:, 1], te2[:, 1], s=2, alpha=0.5, linewidth=0)
    # # axs[1, 1].set_title(f'Test: second CC, Corr: {onp.corrcoef(te1[:, 1], te2[:, 1])[0, 1]:.3f}')
    # #
    # # axs = axs.flatten()
    # # for ax in axs:
    # #     ax.set_xlabel('V1')
    # #     ax.set_ylabel('V2')
    # #
    # # fig.suptitle('CCA w/ diagonal cov matrix. Train/test split in half.')
    # # plt.show()
