import numpy as np
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA


def run_cvPCA(X, train=None, nshuff=5, seed=942, dim='stim'):
    """
    :param X: 2 x neu x time
    :param train: neu x time
    :param nshuff:
    :return:
    """
    np.random.seed(seed)
    n_components = min(1024, X.shape[2])

    ss = np.zeros((nshuff, n_components))
    for k in range(nshuff):
        print(f'cvPCA Iter {k}')

        idx_flip = np.random.rand(X.shape[2]) > 0.5  # Flip stims. Bootstrap 50%.
        X_use = X.copy()
        X_use[0, :, idx_flip] = X[1, :, idx_flip]
        X_use[1, :, idx_flip] = X[0, :, idx_flip]

        if dim == 'stim':
            if train is not None:
                assert train.shape[1] >= n_components  # t
                assert train.shape[0] == X.shape[1]  # neu
                train_ = train

        elif dim == 'V1':  # Transpose X and train. -> PCs have dim
            if train is not None:
                # assert train.shape[2] == X.shape[2]
                V1 = X_use[:, ypos >= 210, :]
                V2 = X_use[:, ypos < 210, :]
                train_ = V2[0, ...].T
                X_use = np.transpose(V1, axes=(0, 2, 1))
            else:
                X_use = X_use[:, ypos >= 210, :]

        elif dim == 'V2':  # Transpose X and train. -> PCs have dim
            if train is not None:
                # assert train.shape[2] == X.shape[2]
                V1 = X_use[:, ypos >= 210, :]
                V2 = X_use[:, ypos < 210, :]
                train_ = V1[0, ...].T
                X_use = np.transpose(V2, axes=(0, 2, 1))
            else:
                X_use = X_use[:, ypos < 210, :]
        else:
            raise Exception('What.')


        if train is None:
            ss[k, :] = _cvPCA(X_use, X_use[0, ...], n_components)
        else:
            idx_other = np.random.rand(train_.shape[1]) > 0.5  # Choose time.
            ss[k, :] = _cvPCA(X_use, train_[:, idx_other], n_components)

    return ss

def _cvPCA(X, train, n_components):
    assert X.shape[1] == train.shape[0]
    model = PCA(n_components=n_components).fit(train.T)  # X = UΣV^T
    # Generate 'super-neurons'
    comp = model.components_.T  # n_components x neu
    # Rotate entire dataset and extract first {n_components} dims, aka low-rank descriptions of neuronal activities.
    # Then calculate inner products between {n_components} stim vectors, aka covariance.
    return np.sum((X[0, ...].T @ comp) * (X[1, ...].T @ comp), axis=0)

def fit_powerlaw(ss, dmin=50, dmax=500):
    def power_law(x, k, α):
        return k * x ** α

    popt, pcov = curve_fit(power_law, np.arange(dmin, dmax), ss[dmin:dmax])
    return popt, pcov