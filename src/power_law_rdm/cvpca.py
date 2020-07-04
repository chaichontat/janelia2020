from __future__ import annotations

from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA

from ..analyzer import Analyzer
from ..spikeloader import SpikeLoader
from .subtract_spont import SubtractSpontAnalyzer


# %%
class cvPCA(Analyzer):

    HYPERPARAMS = ['n_pc', 'n_shuff', 'seed']
    ARRAYS = ['sums_of_squares']
    DATAFRAMES = None

    def __init__(self, n_pc: int = 1024, n_shuff: int = 5, seed: int = 124,
                 sums_of_squares: np.ndarray = None):

        self.n_pc = n_pc
        self.n_shuff = n_shuff
        self.seed = seed

        self.sums_of_squares = sums_of_squares

    def _cvpca_decorator(cvpca_func):
        @wraps(cvpca_func)
        def cvpca_routines(self: cvPCA, X: np.ndarray, *args, **kwargs):
            np.random.seed(self.seed)

            # Clear old inputs.
            self.X = np.empty(0)
            self.Y = np.empty(0)

            # Check inputs.
            assert X.ndim == 3
            assert X.shape[0] == 2

            ss = np.zeros((self.n_shuff, self.n_pc))  # Sums of squares.
            for k in range(self.n_shuff):
                print(f'cvPCA Iter {k + 1}')
                ss[k, :] = cvpca_func(self, X, *args, **kwargs)

            ss = np.mean(ss, axis=0)
            ss /= np.sum(ss)

            # Record
            self.sums_of_squares = ss
            return self.sums_of_squares

        return cvpca_routines

    @_cvpca_decorator
    def run_cvpca(self, X):
        X_swapped = self.swap_idx_between_repeats(X)
        return self._cvPCA(X_swapped, X_swapped)

    @_cvpca_decorator
    def run_cvpca_external_eigvec(self, X, Y):
        assert Y.ndim == 3
        X_swapped = self.swap_idx_between_repeats(X)

        if X.shape[2] != Y.shape[2]:  # Unequal neurons.
            X_use = np.transpose(X_swapped, (0, 2, 1))
            Y_use = np.transpose(self.swap_idx_between_repeats(Y), (0, 2, 1))
        else:
            assert X.shape[2] == Y.shape[2]  # Equal neurons.
            X_use = X_swapped
            Y_use = Y

        return self._cvPCA(X_use, Y_use)

    @staticmethod
    def swap_idx_between_repeats(X):
        idx_flip = np.random.rand(X.shape[1]) > 0.5  # Flip stims. Bootstrap 50%.
        X_use = X.copy()
        X_use[0, idx_flip, :] = X[1, idx_flip, :]
        X_use[1, idx_flip, :] = X[0, idx_flip, :]
        return X_use

    def _cvPCA(self, X, train):
        assert X.shape[1] >= self.n_pc
        assert X.shape[2] >= self.n_pc
        assert X.shape[2] == train.shape[2]

        model = PCA(n_components=self.n_pc, random_state=np.random.RandomState(self.seed)).fit(train[0, ...])  # X = UΣV^T
        comp = model.components_.T
        # Rotate entire dataset and extract first {n_components} dims, aka low-rank descriptions of neuronal activities.
        # Then calculate inner products between {n_components} stim vectors, aka covariance.
        return np.sum((X[0, ...] @ comp) * (X[1, ...] @ comp), axis=0)

    def fit_powerlaw(self, dmin=50, dmax=500):
        dmax = min(dmax, self.n_pc)
        def power_law(x, k, α):
            return k * x ** α

        popt, pcov = curve_fit(power_law, np.arange(dmin, dmax), self.sums_of_squares[dmin:dmax])
        return popt, pcov


def gen_cvpca_format(S_nospont, idx_rep, idx_notrep):
    rep = np.transpose(S_nospont[idx_rep, :], axes=(1, 0, 2))  # 2 x time x neu
    notrep = S_nospont[idx_notrep, :][np.newaxis, ...]
    return rep, notrep


if __name__ == '__main__':
    sns.set()

    loader = SpikeLoader.from_hdf5('tests/data/raw.hdf5')
    spont_model = SubtractSpontAnalyzer().fit(loader.spks, loader.get_idx_spont())
    S_nospont = spont_model.transform(loader.S)


    rep, notrep = gen_cvpca_format(S_nospont, *loader.get_idx_rep(return_onetimers=True))
    ypos = loader.pos['y']

    def cvPCA_traintest(X, Y, ax1, ax2, name=None, name_eigvec=None):
        sss = []
        for i, ax in enumerate([ax1, ax2]):
            cv = cvPCA(n_shuff=2, n_pc=200)
            if i == 0:
                ss = cv.run_cvpca(X)
            else:
                ss = cv.run_cvpca_external_eigvec(X, Y)

            sss.append(ss)
            popt, pcov = cv.fit_powerlaw(dmin=20, dmax=800)

            ax.loglog(ss)
            ax.loglog(popt[0] * np.arange(1, len(ss) + 1) ** popt[1], '--')
            ax.set_title(f'{name} w/ {name_eigvec[i]} eigenvectors, α={popt[1]: 0.3f}')

            ax.set_xlabel('PC dimensions')
            ax.set_ylabel('Variance (cumulative)')

        return sss


    # %% Compare V1 and V2 eigenspectrum decay between repeated and non-repeated stimuli.
    fig, axs = plt.subplots(figsize=(12, 8), nrows=2, ncols=2, dpi=300, constrained_layout=True)
    ss_v1_rep = cvPCA_traintest(rep[:, :, ypos >= 210], notrep[:, :, ypos >= 210], *axs[:, 0],
                                name='V1 rep', name_eigvec=['rep stim', 'non-rep stim'])

    ss_v2_rep = cvPCA_traintest(rep[:, :, ypos < 210], notrep[:, :, ypos < 210], *axs[:, 1],
                                name='V2 rep', name_eigvec=['rep stim', 'non-rep stim'])
    fig.suptitle('cvPCA Eigenspectra of PCs with neuron dims. Comparing eigvecs from rep/non-rep stim.')
    plt.show()

    # %% Compare V1 and V2 eigenspectrum decay between the two regions.
    fig, axs = plt.subplots(figsize=(12, 8), nrows=2, ncols=2, dpi=300, constrained_layout=True)
    ss_v1 = cvPCA_traintest(rep[:, :, ypos >= 210], rep[:, :, ypos < 210], *axs[:, 0],
                            name='V1', name_eigvec=['V1', 'V2'])
    ss_v2 = cvPCA_traintest(rep[:, :, ypos < 210], rep[:, :, ypos >= 210], *axs[:, 1],
                            name='V2', name_eigvec=['V2', 'V1'])
    fig.suptitle('cvPCA Eigenspectra of PCs with stim dims. Comparing eigvecs from V1 and V2.')
    plt.show()

# %%
# def make_percentile(data):
#     return (stats.rankdata(data) / data.size).reshape(data.shape)
#
# from rastermap import Rastermap
# rast = Rastermap(n_components=1, alpha=2.)
# x = rast.fit_transform(S_nospont[:, idx_rep[:, 0]].T)
#
# #%%
# # idxs = [ypos>=210, ypos<210]
# #
# fig, axs = plt.subplots(ncols=2, figsize=(12, 5), dpi=300, constrained_layout=True, clear=True)
# for i in range(2):
#     test = make_percentile(np.corrcoef(S_nospont[:, idx_rep[:, i]].T))
#     test = test[:, x.flatten().argsort()]
#     test = test[x.flatten().argsort(), :]
#     u = axs[i].imshow(test, cmap='twilight_shifted')
#     axs[i].grid(0)
#     axs[i].set_title(f'Repeat {i+1}')
#     fig.colorbar(u, ax=axs[i])
#
# #
# fig.suptitle('RSM, Repeated stim, subtracted out 25 PCs of spont activities. Sorted by Rastermap from Rep 1. Color depicts percentile.')
# fig.savefig('sp_rem.png')
# plt.show()
#
# # #%%
# # from sklearn.manifold import MDS
# # pcoa = MDS(n_components=1, dissimilarity='precomputed')
# # fuck = pcoa.fit_transform(1-make_rdm(S[:, out[:, 0]]))
# #
# # #%%
# # from sklearn.manifold import TSNE
# # model = TSNE(n_components=1, verbose=1)
# # x = model.fit_transform(S[:, out[:, 0]].T)
#
# #%%
# fig, axs = plt.subplots(ncols=2, figsize=(12, 5), dpi=300, constrained_layout=True, clear=True)
# # x=sns.clustermap(np.corrcoef(S[:, out[:, 0]].T))
# for i in range(2):
#     test = make_percentile(np.corrcoef(Sspont.T[:, idxs[i]]))
#     test = test[:, x.flatten().argsort()]
#     test = test[x.flatten().argsort(), :]
#     u = axs[i].imshow(test, cmap='twilight_shifted')
#     axs[i].grid(0)
#     axs[i].set_title(f'V{i+1}')
#     fig.colorbar(u, ax=axs[i])
# #
# fig.suptitle('RSM, Spontaneous frames, sorted by 1D Rastermap. Color depicts percentile.')
# plt.savefig('spont.png')
# plt.show()
