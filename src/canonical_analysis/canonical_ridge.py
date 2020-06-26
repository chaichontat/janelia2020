import pickle
from pathlib import Path
from typing import Tuple, Union

import jax.numpy as np
import matplotlib.pyplot as plt
import numpy as onp
import seaborn as sns
from jax.numpy.linalg import inv, cholesky
from sklearn.decomposition import PCA

from ..analyzer import Analyzer
from ..receptive_field.rf import ReceptiveField
from ..spikeloader import SpikeLoader

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
        self.X_ref, self.Y_ref = np.empty(0), np.empty(0)
        self.coef = np.empty(0)

    def fit(self, X: Arrays, Y: Arrays):
        assert X.shape[0] == Y.shape[0]
        self.X_ref, self.Y_ref = X, Y
        K = self._calc_canon_mat(X, Y)
        model = PCA(self.n).fit(K)  # Randomized SVD.

        # Based on X = UΣV^T.
        self.V = model.components_.T
        self.singular_values = self.Σ = np.diag(model.singular_values_)
        self.U = K @ self.V @ np.linalg.inv(self.Σ)

        self.coef = self.calc_canon_coef(X, Y)
        return self

    def fit_transform(self, X: Arrays, Y: Arrays) -> Tuple[np.DeviceArray, np.DeviceArray]:
        self.X_ref, self.Y_ref = X, Y
        self.fit(X, Y)
        return self.transform(X, Y)

    def transform(self, X: Arrays, Y: Arrays) -> Tuple[np.DeviceArray, np.DeviceArray]:
        return X @ self.U, Y @ self.V

    def _calc_canon_mat(self, X: Arrays, Y: Arrays) -> np.DeviceArray:
        X, Y = np.asarray(X), np.asarray(Y)

        X -= np.mean(X, axis=0)
        Y -= np.mean(Y, axis=0)
        K = inv(cholesky((1 - self.λx) * np.cov(X, rowvar=False) + self.λx * np.eye(X.shape[1]))) @ \
            (X.T @ Y) / (X.T.shape[0] * Y.shape[1]) @ \
            inv(cholesky((1 - self.λy) * np.cov(Y, rowvar=False) + self.λy * np.eye(Y.shape[1])))

        return K

    def calc_canon_coef(self, X: Arrays, Y: Arrays) -> np.DeviceArray:
        X_p, Y_p = self.transform(X, Y)
        return np.array([onp.corrcoef(X_p[:, i], Y_p[:, i])[0, 1] for i in range(self.n)])

    def subtract_canon_comp(self, X: Arrays, Y: Arrays) -> Tuple[np.DeviceArray, np.DeviceArray]:
        X_comp, Y_comp = self.transform(X, Y)
        return X - (X_comp @ self.U.T), Y - (Y_comp @ self.V.T)


if __name__ == '__main__':
    sns.set()
    loader = SpikeLoader()

    print('Running CCA.')
    V1, V2 = loader.S[:, loader.ypos >= 210], loader.S[:, loader.ypos < 210]
    cca = CanonicalRidge().fit(V1, V2)
    Path('cc.pk').write_bytes(pickle.dumps(cca, protocol=5))

    fig, ax = plt.subplots()
    ax.plot(cca.coef)
    ax.set_title('Canonical Correlation Coefficients')
    ax.set_xlabel('CCs')
    ax.set_ylabel('Pearson\'s r')

    V1s, V2s = cca.subtract_canon_comp(cca.X_ref, cca.Y_ref)
    rf_v1 = ReceptiveField(loader).fit_neuron(loader.X, V1s)
    rf_v1.plot_rf()
    rf_v2 = ReceptiveField(loader).fit_neuron(loader.X, V2s)
    rf_v2.plot_rf()
