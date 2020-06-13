
import scipy.ndimage as ndi
from scipy.stats import zscore

# import jax.numpy as np
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
sns.set()

#%% Pre-processing
print('Loading data.')
raw = np.load('superstim.npz')
for name in raw.files:
    globals()[name] = raw[name]

S = zscore(spks[:, frame_start], axis=0)  # neu x time

#%% Remove spontaneous dimensions.
idx_spont = np.array(list(set(range(np.max(frame_start + 1))) - set(frame_start)), dtype=np.int32)
S_spont = zscore(spks[:, idx_spont], axis=1)  # neu x time

pca_spont = PCA(n_components=25)
pca_spont.fit(S_spont.T)  # time x neu
pcs_spont = pca_spont.components_  # n_components x neu

proj_spont = S.T @ pcs_spont.T  # time x n_components
S_nospont = zscore((S.T - (proj_spont @ pcs_spont)).T, axis=1)  # neu x time


#%% Get index of repeating images.
unq, unq_cnt = np.unique(istim, return_counts=True)
repeating_img = unq[np.argwhere(unq_cnt == 2)]  # idx of repeating img
idx_rep = np.zeros([len(repeating_img), 2], dtype=np.int32)  # First and second occurrence
for i in range(len(repeating_img)):
    idx_rep[i, :] = np.where(istim == repeating_img[i])[0]

#%%
for_cvpca = np.transpose(S_nospont[:, idx_rep], axes=(2, 1, 0))

from utils import *
ss = shuff_cvPCA(for_cvpca)
fuck = np.mean(ss, axis=0)
fuck /= np.sum(fuck)

from scipy.optimize import curve_fit

def func(x, k, α):
    return k * x ** α
#%%
popt, pcov = curve_fit(func, np.arange(50, 500), fuck[50:500])
fig, ax = plt.subplots(dpi=300)
ax.loglog(fuck)
ax.loglog(popt[0] * np.arange(1, len(fuck)+1)**popt[1], '--')
ax.set_title(f'Eigenspectrum from 5-fold cvPCA, α={popt[1]}')
ax.set_xlabel('PC dimensions')
ax.set_ylabel('Variance (cumulative)')
plt.show()

#%%
def make_percentile(data):
    return (stats.rankdata(data) / data.size).reshape(data.shape)

from rastermap import Rastermap
rast = Rastermap(n_components=1, alpha=2.)
x = rast.fit_transform(S_nospont[:, idx_rep[:, 0]].T)

#%%
# idxs = [ypos>=210, ypos<210]
#
fig, axs = plt.subplots(ncols=2, figsize=(12, 5), dpi=300, constrained_layout=True, clear=True)
for i in range(2):
    test = make_percentile(np.corrcoef(S_nospont[:, idx_rep[:, i]].T))
    test = test[:, x.flatten().argsort()]
    test = test[x.flatten().argsort(), :]
    u = axs[i].imshow(test, cmap='bwr')
    axs[i].grid(0)
    axs[i].set_title(f'Repeat {i+1}')
    fig.colorbar(u, ax=axs[i])

#
fig.suptitle('RSM, Repeated stim, subtracted out 25 PCs of spont activities. Sorted by Rastermap from Rep 1. Color depicts percentile.')
fig.savefig('sp_rem.png')
plt.show()

# #%%
# from sklearn.manifold import MDS
# pcoa = MDS(n_components=1, dissimilarity='precomputed')
# fuck = pcoa.fit_transform(1-make_rdm(S[:, out[:, 0]]))
#
# #%%
# from sklearn.manifold import TSNE
# model = TSNE(n_components=1, verbose=1)
# x = model.fit_transform(S[:, out[:, 0]].T)

#%%
fig, axs = plt.subplots(ncols=2, figsize=(12, 5), dpi=300, constrained_layout=True, clear=True)
# x=sns.clustermap(np.corrcoef(S[:, out[:, 0]].T))
for i in range(2):
    test = make_percentile(np.corrcoef(Sspont.T[:, idxs[i]]))
    test = test[:, x.flatten().argsort()]
    test = test[x.flatten().argsort(), :]
    u = axs[i].imshow(test, cmap='bwr')
    axs[i].grid(0)
    axs[i].set_title(f'V{i+1}')
    fig.colorbar(u, ax=axs[i])
#
fig.suptitle('RSM, Spontaneous frames, sorted by 1D Rastermap. Color depicts percentile.')
plt.savefig('spont.png')
plt.show()