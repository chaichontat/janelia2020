import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA

from utils_powerlaw import Data


class cvPCA(Data):
    def prep_data(self):
        # %% Get indices of repeating images.
        unq, unq_cnt = np.unique(self.istim, return_counts=True)
        idx_firstrep = unq[np.argwhere(unq_cnt == 2)]  # idx of repeating img
        idx_rep = np.zeros([len(idx_firstrep), 2], dtype=np.int32)  # First and second occurrences
        for i in range(len(idx_firstrep)):
            idx_rep[i, :] = np.where(self.istim == idx_firstrep[i])[0]
        assert unq.size + idx_firstrep.size == self.istim.size

        # %% cvPCA with train/test split.
        for_cvpca = np.transpose(self.S_corr[:, idx_rep], axes=(2, 0, 1))  # 2 x neu x time
        idx_notrep = np.where(np.isin(np.arange(len(self.istim)), np.array(idx_rep).flatten(), invert=True))[0]
        assert idx_notrep.size + idx_rep.size == self.istim.size

        return for_cvpca, idx_rep, idx_notrep

    def run(self, X, train=None, nshuff=5, seed=942, dim='stim'):
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
                    V1 = X_use[:, self.ypos >= 210, :]
                    V2 = X_use[:, self.ypos < 210, :]
                    train_ = V2[0, ...].T
                    X_use = np.transpose(V1, axes=(0, 2, 1))
                else:
                    X_use = X_use[:, self.ypos >= 210, :]

            elif dim == 'V2':  # Transpose X and train. -> PCs have dim
                if train is not None:
                    # assert train.shape[2] == X.shape[2]
                    V1 = X_use[:, self.ypos >= 210, :]
                    V2 = X_use[:, self.ypos < 210, :]
                    train_ = V1[0, ...].T
                    X_use = np.transpose(V2, axes=(0, 2, 1))
                else:
                    X_use = X_use[:, self.ypos < 210, :]
            else:
                raise Exception('What.')

            if train is None:
                ss[k, :] = self._cvPCA(X_use, X_use[0, ...], n_components)
            else:
                idx_other = np.random.rand(train_.shape[1]) > 0.5  # Choose time.
                ss[k, :] = self._cvPCA(X_use, train_[:, idx_other], n_components)

        ss = np.mean(ss, axis=0)
        ss /= np.sum(ss)
        return ss

    @staticmethod
    def _cvPCA(X, train, n_components):
        assert X.shape[1] == train.shape[0]
        model = PCA(n_components=n_components).fit(train.T)  # X = UΣV^T
        # Generate 'super-neurons'
        comp = model.components_.T  # n_components x neu
        # Rotate entire dataset and extract first {n_components} dims, aka low-rank descriptions of neuronal activities.
        # Then calculate inner products between {n_components} stim vectors, aka covariance.
        return np.sum((X[0, ...].T @ comp) * (X[1, ...].T @ comp), axis=0)

    @staticmethod
    def fit_powerlaw(ss, dmin=50, dmax=500):
        def power_law(x, k, α):
            return k * x ** α

        popt, pcov = curve_fit(power_law, np.arange(dmin, dmax), ss[dmin:dmax])
        return popt, pcov


if __name__ == '__main__':
    sns.set()
    cv = cvPCA()


    def cvPCA_traintest(X, train, ax1, ax2, name=None, curr=None, dim='stim'):
        sss = []
        for i, ax in enumerate([ax1, ax2]):
            if i == 0:
                ss = cv.run(X, train, nshuff=2, dim=dim)
            else:
                ss = cv.run(X, nshuff=2, dim=dim)

            sss.append(ss)
            popt, pcov = cv.fit_powerlaw(ss, dmin=20, dmax=800)

            ax.loglog(ss)
            ax.loglog(popt[0] * np.arange(1, len(ss) + 1) ** popt[1], '--')
            ax.set_title(f'{name} w/ {curr[i]} eigenvectors, α={popt[1]: 0.3f}')

            ax.set_xlabel('PC dimensions')
            ax.set_ylabel('Variance (cumulative)')

        return sss


    X, idx_rep, idx_notrep = cv.prep_data()
    # %%
    fig, axs = plt.subplots(figsize=(12, 8), nrows=2, ncols=2, dpi=300, constrained_layout=True)
    ss_v1_rep = cvPCA_traintest(X[:, cv.ypos >= 210, :], cv.S_corr[:, idx_notrep][cv.ypos >= 210, :], *axs[:, 0],
                                name='V1 rep', curr=['non-rep stim', 'rep stim'])
    ss_v2_rep = cvPCA_traintest(X[:, cv.ypos < 210, :], cv.S_corr[:, idx_notrep][cv.ypos < 210, :], *axs[:, 1],
                                name='V2 rep', curr=['non-rep stim', 'rep stim'])
    fig.suptitle('cvPCA Eigenspectra of PCs with neuron dims. Comparing eigvecs from rep/non-rep stim.')
    plt.show()

    fig, axs = plt.subplots(figsize=(12, 8), nrows=2, ncols=2, dpi=300, constrained_layout=True)
    ss_v1 = cvPCA_traintest(X, 'hi', *axs[:, 0], name='V1', dim='V1', curr=['V2', 'V1'])
    ss_v2 = cvPCA_traintest(X, 'hi', *axs[:, 1], name='V2', dim='V2', curr=['V1', 'V2'])
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
