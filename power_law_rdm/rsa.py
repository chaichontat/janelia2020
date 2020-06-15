import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

from utils_powerlaw import *

sns.set()


# %% Pre-processing
print('Loading data.')
raw = np.load('../superstim.npz')
for name in raw.files:
    globals()[name] = raw[name]

assert len(frame_start) == len(istim)
S = zscore(spks[:, frame_start], axis=1)  # neu x time


# %% Remove spontaneous PCs.
idx_spont = np.where(np.isin(np.arange(np.max(frame_start) + 1), frame_start, assume_unique=True, invert=True))[0]  # Invert indices.
assert idx_spont.size + frame_start.size == spks.shape[1]

S_spont = zscore(spks[:, idx_spont], axis=1)  # neu x time

pca_spont = PCA(n_components=25).fit(S_spont.T)  # time x neu ; pcs are 'superneurons'
pcs_spont = pca_spont.components_  # n_components x neu

proj_spont = S.T @ pcs_spont.T  # neu x n_components
S_corr = (S.T - proj_spont @ pcs_spont).T  # neu x time
S_corr -= np.mean(S_corr, axis=1)[:, np.newaxis]


# %% Get indices of repeating images.
unq, unq_cnt = np.unique(istim, return_counts=True)
idx_firstrep = unq[np.argwhere(unq_cnt == 2)]  # idx of repeating img
idx_rep = np.zeros([len(idx_firstrep), 2], dtype=np.int32)  # First and second occurrences
for i in range(len(idx_firstrep)):
    idx_rep[i, :] = np.where(istim == idx_firstrep[i])[0]
assert unq.size + idx_firstrep.size == istim.size


# %% cvPCA with train/test split.
for_cvpca = np.transpose(S_corr[:, idx_rep], axes=(2, 0, 1))  # 2 x neu x time
idx_notrep = np.where(np.isin(np.arange(len(istim)), np.array(idx_rep).flatten(), invert=True))[0]
assert idx_notrep.size + idx_rep.size == istim.size

# ss = run_cvPCA(for_cvpca, S_corr[:, idx_notrep], nshuff=2)
# ss = np.mean(ss, axis=0)
# ss /= np.sum(ss)
#
# popt, pcov = fit_powerlaw(ss)
#
# fig, ax = plt.subplots(dpi=300)
# ax.loglog(ss)
# ax.loglog(popt[0] * np.arange(1, len(ss) + 1) ** popt[1], '--')
# ax.set_title(f'Eigenspectrum from 10-fold cvPCA w/ test eigenvectors, α={popt[1]: 0.3f}')
# ax.set_xlabel('PC dimensions')
# ax.set_ylabel('Variance (cumulative)')
# plt.show()

#%%
def cvPCA_traintest(X, train, ax1, ax2, name=None, dim='stim'):
    sss = []
    for i, ax in enumerate([ax1, ax2]):
        if i == 0:
            curr = 'train'
            ss = run_cvPCA(X, train, nshuff=5, dim=dim)
        else:
            curr = 'test'
            ss = run_cvPCA(X, nshuff=5, dim=dim)

        sss.append(ss)
        ss = np.mean(ss, axis=0)
        ss /= np.sum(ss)
        popt, pcov = fit_powerlaw(ss)

        ax.loglog(ss)
        ax.loglog(popt[0] * np.arange(1, len(ss) + 1) ** popt[1], '--')
        ax.set_title(f'{name} w/ {curr} eigenvectors, α={popt[1]: 0.3f}')

        ax.set_xlabel('PC dimensions')
        ax.set_ylabel('Variance (cumulative)')

    return sss

# fig, axs = plt.subplots(figsize=(12,8), nrows=2, ncols=2, dpi=300, constrained_layout=True)
# ss_v1 = cvPCA_traintest(for_cvpca[:, ypos >= 210, :], S_corr[:, idx_notrep][ypos >= 210, :], *axs[:, 0], name='V1')
# ss_v2 = cvPCA_traintest(for_cvpca[:, ypos < 210, :], S_corr[:, idx_notrep][ypos < 210, :], *axs[:, 1], name='V2')
# fig.suptitle('Eigenspectra from cvPCA')
# plt.show()


fig, axs = plt.subplots(figsize=(12,8), nrows=2, ncols=2, dpi=300, constrained_layout=True)
ss_v1 = cvPCA_traintest(for_cvpca[:, ypos >= 210, :], for_cvpca[:, ypos < 210, :], *axs[:, 0], name='V1', dim='neu')
ss_v2 = cvPCA_traintest(for_cvpca[:, ypos < 210, :], for_cvpca[:, ypos >= 210, :], *axs[:, 1], name='V2', dim='neu')
fig.suptitle('Eigenspectra of stim PCs from cvPCA, train/test from V1 or V2')
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
