from typing import Tuple, Union

import jax.numpy as np
import matplotlib.pyplot as plt
import numpy as onp
from jax.numpy.linalg import inv, cholesky
from sklearn.decomposition import PCA

from ..analyzer import Analyzer
from ..receptive_field.rf import ReceptiveField
from ..spikeloader import SpikeLoader
from ..utils.io import hdf5_save

Arrays = Union[np.DeviceArray, onp.ndarray]


class CanonicalRidge(Analyzer):
    """
    Class for canonical correlation analysis with ridge regularization.
    See http://www2.imm.dtu.dk/pubdb/edoc/imm4981.pdf

    Uses JAX for faster numerical computation.
    Based on the scikit-learn Estimator API.

    """

    params = ['n', 'lambda_x', 'lambda_y', 'seed']
    arrs = ['V', 'U', 'singular_values', 'coef', 'transformed_U', 'transformed_V']
    dfs = None

    def __init__(self, n: int = 25, lambda_x: float = 0.85, lambda_y: float = 0.85, seed: int = 87,
                 V=None, singular_values=None, U=None, coef=None, transformed_U=None, transformed_V=None):
        self.n = n
        self.lambda_x = lambda_x
        self.lambda_y = lambda_y
        self.seed = seed

        self.V, self.singular_values, self.U = V, singular_values, U
        self.Σ = self.singular_values
        self.coef = coef
        self.transformed_U, self.transformed_V = transformed_U, transformed_V


    def fit(self, X: Arrays, Y: Arrays):
        assert X.shape[0] == Y.shape[0]
        K = self._calc_canon_mat(X, Y)
        model = PCA(self.n, random_state=onp.random.RandomState(self.seed)).fit(K)  # Randomized SVD.

        # Based on X = UΣV^T.
        self.V = model.components_.T
        self.singular_values = self.Σ = np.diag(model.singular_values_)
        self.U = K @ self.V @ np.linalg.inv(self.Σ)

        self.coef = self.calc_canon_coef(X, Y)
        return self

    def fit_transform(self, X: Arrays, Y: Arrays) -> Tuple[np.DeviceArray, np.DeviceArray]:
        self.fit(X, Y)
        return self.transform(X, Y)

    def transform(self, X: Arrays, Y: Arrays) -> Tuple[np.DeviceArray, np.DeviceArray]:
        self.transformed_U, self.transformed_V = X @ self.U, Y @ self.V
        return self.transformed_U, self.transformed_V

    def _calc_canon_mat(self, X: Arrays, Y: Arrays) -> np.DeviceArray:
        X, Y = np.asarray(X), np.asarray(Y)

        X -= np.mean(X, axis=0)
        Y -= np.mean(Y, axis=0)
        K = inv(cholesky((1 - self.lambda_x) * np.cov(X, rowvar=False) + self.lambda_x * np.eye(X.shape[1]))) @ \
            (X.T @ Y) / (X.T.shape[0] * Y.shape[1]) @ \
            inv(cholesky((1 - self.lambda_y) * np.cov(Y, rowvar=False) + self.lambda_y * np.eye(Y.shape[1])))

        return K

    def calc_canon_coef(self, X: Arrays, Y: Arrays) -> np.DeviceArray:
        if self.transformed_U is None or self.transformed_V is None:
            self.transform(X, Y)
        X_p, Y_p = self.transformed_U, self.transformed_V
        return np.array([onp.corrcoef(X_p[:, i], Y_p[:, i])[0, 1] for i in range(self.n)])

    def subtract_canon_comp(self, X: Arrays, Y: Arrays) -> Tuple[np.DeviceArray, np.DeviceArray]:
        if self.transformed_U is None or self.transformed_V is None:
            self.transform(X, Y)
        return X - (self.transformed_U @ self.U.T), Y - (self.transformed_V @ self.V.T)


def make_regression_truth():
    loader = SpikeLoader.from_hdf5('tests/data/processed.hdf5')
    print('Running CCA.')
    V1, V2 = loader.S[:, loader.pos['y'] >= 210], loader.S[:, loader.pos['y'] < 210]
    cca = CanonicalRidge().fit(V1, V2)

    fig, ax = plt.subplots()
    ax.plot(cca.coef)
    ax.set_title('Canonical Correlation Coefficients')
    ax.set_xlabel('CCs')
    ax.set_ylabel('Pearson\'s r')

    V1s, V2s = cca.subtract_canon_comp(V1, V2)
    rf_v1 = ReceptiveField(loader.img_dim).fit_pc(loader.imgs_stim, V1s)
    rf_v1.plot_rf()
    rf_v2 = ReceptiveField(loader.img_dim).fit_pc(loader.imgs_stim, V2s)
    rf_v2.plot_rf()

    hdf5_save('tests/data/regression_test_data.hdf5', 'CanonicalRidge',
              arrs={'V1': rf_v1.rf_, 'V2': rf_v2.rf_}, append=True, overwrite_node=True)
