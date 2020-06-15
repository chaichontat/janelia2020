import numpy as np
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA


def run_cvPCA(X, train=None, nshuff=5, seed=942):
    """
    :param X: 2 x neu x time
    :param train: neu x time
    :param nshuff:
    :return:
    """
    np.random.seed(seed)
    n_components = min(1024, X.shape[2])

    if train is not None:
        assert train.shape[1] >= n_components  # t
        assert X.shape[1] == train.shape[0]  # neu

    ss = np.zeros((nshuff, n_components))
    for k in range(nshuff):
        print(f'cvPCA Iter {k}')

        idx_flip = np.random.rand(X.shape[2]) > 0.5  # Bootstrap 50%.
        X_use = X.copy()
        X_use[0, :, idx_flip] = X[1, :, idx_flip]
        X_use[1, :, idx_flip] = X[0, :, idx_flip]

        if train is None:
            ss[k, :] = _cvPCA(X_use, X_use[0, ...], n_components)
        else:
            idx_other = np.random.rand(train.shape[1]) > 0.5  # Choose time.
            ss[k, :] = _cvPCA(X_use, train[:, idx_other], n_components)

    return ss


def _cvPCA(X, train, nc):
    model = PCA(n_components=nc).fit(train)  # neu x stim ; pcs are 'supertime'

    xproj = train @ (model.components_.T / model.singular_values_)
    cproj0 = X[0, ...].T @ xproj  # Get component in that direction.
    cproj1 = X[1, ...].T @ xproj

    return np.sum(cproj0 * cproj1, axis=0)  # marginalize time


def fit_powerlaw(ss, dmin=50, dmax=500):
    def power_law(x, k, α):
        return k * x ** α

    popt, pcov = curve_fit(power_law, np.arange(dmin, dmax), ss[dmin:dmax])
    return popt, pcov