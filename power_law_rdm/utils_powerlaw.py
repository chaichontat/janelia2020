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
        idx_other = np.random.choice(np.arange(another.shape[1]), size=nc, replace=False)  # Choose time.

        idx_X = np.random.choice(np.arange(X.shape[2]), size=nc, replace=False)  # Choose time.
        flip = np.random.choice([0, 1])

        X_use = np.zeros((2, X.shape[1], nc))
        X_use[flip, :, :] = X[0, :, idx_X].T  # Advanced indexing comes first.
        X_use[1 - flip, :, :] = X[1, :, idx_X].T

        ss[k] = cvPCA(X_use, another[:, idx_other])

    return ss


def cvPCA(X, another):
    model = PCA(n_components=X.shape[2]).fit(another)  # neu x stim

    xproj = X[0, ...] @ (model.components_.T / model.singular_values_)
    cproj0 = X[0, ...].T @ xproj
    cproj1 = X[1, ...].T @ xproj

    return np.sum(cproj0 * cproj1, axis=0)


def fit_powerlaw(ss, dmin=50, dmax=500):
    def power_law(x, k, α):
        return k * x ** α

    popt, pcov = curve_fit(power_law, np.arange(dmin, dmax), ss[dmin:dmax])
    return popt, pcov

#
# def shuff_cvPCA(X, another, nshuff=5, seed=942):
#     """
#     :param X: 2 x stimuli x neurons
#     :param nshuff:
#     :return:
#     """
#     np.random.seed(seed)
#     nc = np.min(1024, X.shape[1])
#     ss = np.zeros((nshuff, nc))
#
#     for k in range(nshuff):
#         idxs = np.random.choice([0, 1], size=nc).astype(np.bool)  # np.random.rand(X.shape[1]) > 0.5
#         flip = np.random.choice([0, 1])
#
#         X0 = np.zeros([2, nc, X.shape[2]])
#         X0[flip, idxs] = X[0, idxs]
#         X0[1 - flip, idxs] = X[1, idxs]
#         ss[k] = cvPCA(X0, another, nc)
#     return ss
#
#
# def cvPCA(X, another, nc):
#     ''' X is 2 x stimuli x neurons '''
#     pca = PCA(n_components=nc).fit(X[0].T)
#     u = pca.components_.T
#     sv = pca.singular_values_
#
#     xproj = X[0].T @ (u / sv)
#     cproj0 = X[0] @ xproj
#     cproj1 = X[1] @ xproj
#     ss = (cproj0 * cproj1).sum(axis=0)
#     return ss
