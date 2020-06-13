import numpy as np
from sklearn.decomposition import PCA


def shuff_cvPCA(X, nshuff=5, seed=942):
    """
    :param X: 2 x stimuli x neurons
    :param nshuff:
    :return:
    """
    np.random.seed(seed)
    nc = np.min(1024, X.shape[1])
    ss = np.zeros((nshuff, nc))

    for k in range(nshuff):
        idxs = np.random.choice([0, 1], size=nc).astype(np.bool)  # np.random.rand(X.shape[1]) > 0.5
        flip = np.random.choice([0, 1])

        X0 = np.zeros([2, nc, X.shape[2]])
        X0[flip, idxs] = X[0, idxs]
        X0[1 - flip, idxs] = X[1, idxs]
        ss[k] = cvPCA(X0, nc)
    return ss


def cvPCA(X, nc):
    ''' X is 2 x stimuli x neurons '''
    pca = PCA(n_components=nc).fit(X[0].T)
    u = pca.components_.T
    sv = pca.singular_values_

    xproj = X[0].T @ (u / sv)
    cproj0 = X[0] @ xproj
    cproj1 = X[1] @ xproj
    ss = (cproj0 * cproj1).sum(axis=0)
    return ss