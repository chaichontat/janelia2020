import numpy as np
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA


def shuff_cvPCA(X, another, nshuff=5, seed=942):
    """
    :param X: 2 x neu x time
    :param another: neu x time
    :param nshuff:
    :return:
    """
    np.random.seed(seed)
    nc = min(1024, X.shape[2])
    assert another.shape[1] >= nc  # t
    assert X.shape[1] == another.shape[0]  # neu

    ss = np.zeros((nshuff, nc))
    for k in range(nshuff):
        print(f'cvPCA Iter {k}')
        idx_other = np.random.rand(another.shape[1]) > 0.5  # Choose time.

        idx_flip = np.random.rand(X.shape[2]) > 0.5  # Bootstrap 50%.
        X_use = X.copy()
        X_use[0, :, idx_flip] = X[1, :, idx_flip]
        X_use[1, :, idx_flip] = X[0, :, idx_flip]

        ss[k, :] = cvPCA(X_use, another[:, idx_other], nc)

    return ss


def cvPCA(X, another, nc):
    model = PCA(n_components=nc).fit(another)  # neu x stim ; pcs are 'supertime'

    xproj = another @ (model.components_.T / model.singular_values_)
    cproj0 = X[0, ...].T @ xproj  # Get component in that direction.
    cproj1 = X[1, ...].T @ xproj

    return np.sum(cproj0 * cproj1, axis=0)  # marginalize time


def fit_powerlaw(ss, dmin=50, dmax=500):
    def power_law(x, k, α):
        return k * x ** α

    popt, pcov = curve_fit(power_law, np.arange(dmin, dmax), ss[dmin:dmax])
    return popt, pcov