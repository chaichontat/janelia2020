from __future__ import annotations

from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA

from ..analyzer import SubtractSpontAnalyzer
from ..spikeloader import SpikeStimLoader


# %%
class cvPCA(SubtractSpontAnalyzer):
    def __init__(self, loader: SpikeStimLoader, n_spont_pc: int = 25,
                 n_cvpc: int = 1024, n_shuff: int = 5, seed: int = 124):
        super().__init__(loader, n_spont_pc)
        self.frame_start, self.istim = self.loader.frame_start, self.loader.istim

        self.n_cvpc = n_cvpc
        self.n_shuff = n_shuff
        self.seed = seed

        self.X = np.empty(0)
        self.Y = np.empty(0)
        self.sums_of_squares = np.empty(0)

    def gen_cvpca_format(self, S_nospont=None):
        if S_nospont is not None:
            self.S_nospont = S_nospont

        idx_rep, idx_notrep = self.get_repeating_idx()

        rep = np.transpose(S_nospont[idx_rep, :], axes=(1, 0, 2))  # 2 x time x neu
        notrep = S_nospont[idx_notrep, :][np.newaxis, ...]
        return rep, notrep

    def get_repeating_idx(self):
        # Get indices of repeating images.
        unq, unq_cnt = np.unique(self.istim, return_counts=True)
        idx_firstrep = unq[np.argwhere(unq_cnt == 2)]  # idx of repeating img
        idx_rep = np.zeros([len(idx_firstrep), 2], dtype=np.int32)  # First and second occurrences
        for i in range(len(idx_firstrep)):
            idx_rep[i, :] = np.where(self.istim == idx_firstrep[i])[0]
        assert unq.size + idx_firstrep.size == self.istim.size

        idx_notrep = np.where(np.isin(np.arange(len(self.istim)), np.array(idx_rep).flatten(), invert=True))[0]
        assert idx_notrep.size + idx_rep.size == self.istim.size

        return idx_rep, idx_notrep

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

            ss = np.zeros((self.n_shuff, self.n_cvpc))  # Sums of squares.
            for k in range(self.n_shuff):
                print(f'cvPCA Iter {k + 1}')
                ss[k, :] = cvpca_func(self, X, *args, **kwargs)

            ss = np.mean(ss, axis=0)
            ss /= np.sum(ss)

            # Record
            self.X = X
            self.sums_of_squares = ss
            return self.sums_of_squares

        return cvpca_routines

    @_cvpca_decorator
    def run_cvpca(self, X):
        X_swapped = self.swap_idx_between_repeats(X)
        return self._cvPCA(X_swapped, X_swapped, self.n_cvpc)

    @_cvpca_decorator
    def run_cvpca_external_eigvec(self, X, Y):
        self.Y = Y
        assert Y.ndim == 3
        X_swapped = self.swap_idx_between_repeats(X)

        if X.shape[2] != Y.shape[2]:  # Unequal neurons.
            X_use = np.transpose(X_swapped, (0, 2, 1))
            Y_use = np.transpose(self.swap_idx_between_repeats(Y), (0, 2, 1))
        else:
            assert X.shape[2] == Y.shape[2]  # Equal neurons.
            X_use = X_swapped
            Y_use = Y

        return self._cvPCA(X_use, Y_use, self.n_cvpc)

    @staticmethod
    def swap_idx_between_repeats(X):
        idx_flip = np.random.rand(X.shape[1]) > 0.5  # Flip stims. Bootstrap 50%.
        X_use = X.copy()
        X_use[0, idx_flip, :] = X[1, idx_flip, :]
        X_use[1, idx_flip, :] = X[0, idx_flip, :]
        return X_use

    @staticmethod
    def _cvPCA(X, train, n_cvpc):
        assert X.shape[1] >= n_cvpc
        assert X.shape[2] >= n_cvpc
        assert X.shape[2] == train.shape[2]

        model = PCA(n_components=n_cvpc).fit(train[0, ...])  # X = UΣV^T
        comp = model.components_.T
        # Rotate entire dataset and extract first {n_components} dims, aka low-rank descriptions of neuronal activities.
        # Then calculate inner products between {n_components} stim vectors, aka covariance.
        return np.sum((X[0, ...] @ comp) * (X[1, ...] @ comp), axis=0)

    @staticmethod
    def fit_powerlaw(ss, dmin=50, dmax=500):
        def power_law(x, k, α):
            return k * x ** α

        popt, pcov = curve_fit(power_law, np.arange(dmin, dmax), ss[dmin:dmax])
        return popt, pcov


if __name__ == '__main__':
    sns.set()

    loader = SpikeStimLoader()
    cv = cvPCA(loader, n_shuff=2)
    rep, notrep = cv.gen_cvpca_format(cv.S_nospont)
    ypos = cv.loader.ypos


    def cvPCA_traintest(X, Y, ax1, ax2, name=None, name_eigvec=None):
        sss = []
        for i, ax in enumerate([ax1, ax2]):
            if i == 0:
                ss = cv.run_cvpca(X)
            else:
                ss = cv.run_cvpca_external_eigvec(X, Y)

            sss.append(ss)
            popt, pcov = cv.fit_powerlaw(ss, dmin=20, dmax=800)

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
                            name='V1', name_eigvec=['V2', 'V1'])
    ss_v2 = cvPCA_traintest(rep[:, :, ypos < 210], rep[:, :, ypos >= 210], *axs[:, 1],
                            name='V1', name_eigvec=['V2', 'V1'])
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
#     u = axs[i].imshow(test, cmap='bwr')
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
#     u = axs[i].imshow(test, cmap='bwr')
#     axs[i].grid(0)
#     axs[i].set_title(f'V{i+1}')
#     fig.colorbar(u, ax=axs[i])
# #
# fig.suptitle('RSM, Spontaneous frames, sorted by 1D Rastermap. Color depicts percentile.')
# plt.savefig('spont.png')
# plt.show()
