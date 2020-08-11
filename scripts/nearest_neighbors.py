import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from src.gabor_analysis.gabor_fit import GaborFit
from src.spikeloader import SpikeLoader

sns.set()

"""Nearest neighbor analysis between V1 and V2."""

loader = SpikeLoader.from_hdf5()
gabor = GaborFit.from_hdf5("data/gabor.hdf5")
V2 = np.argwhere(
    (gabor.params_fit[:, -2] > -10) & (loader.pos["y"].to_numpy() <= 200)
).squeeze()  # x-gabor pos
V1 = np.argwhere((gabor.params_fit[:, -2] > -10) & (loader.pos["y"].to_numpy() > 200)).squeeze()

V1 = sorted(np.random.RandomState(seed=439).choice(V1, size=len(V2), replace=False))
assert len(V2) == len(V1)
assert np.isin(V1, V2).sum() == 0

#%%
n_pc = 50
S = loader.S
S = np.hstack([loader.S[:, V1], loader.S[:, V2]])
cutoff = len(V2)

S_hat = PCA(n_components=n_pc, random_state=np.random.RandomState(58)).fit_transform(S.T)

nn = NearestNeighbors(n_neighbors=100).fit(S_hat)

dist, idx = nn.kneighbors(S_hat, n_neighbors=100)


from cuml.manifold import TSNE

V1inV1 = np.sum(idx[:cutoff, :] < cutoff, axis=1)
V2inV2 = np.sum(idx[cutoff:, :] >= cutoff, axis=1)
#
# V1inV1 = np.sum(np.isin(idx[V1, :], V1), axis=1)
# V2inV2 = np.sum(np.isin(idx[V2, :], V2), axis=1)

plt.rcParams["figure.dpi"] = 300
sns.distplot(V1inV1, label="V1inV1", kde=False, norm_hist=True, bins=range(0, 100, 5))
sns.distplot(V2inV2, label="V2inV2", kde=False, norm_hist=True, bins=range(0, 100, 5))
plt.title(
    "Numbers of 100-nearest neighbor of the same type. \nEqual-ish retinotopy ($x>-10$) and equal number of neurons from V1 and V2"
)

plt.legend()
plt.show()
#
#%%
tsne = TSNE(n_components=2, perplexity=40)
X_hat = tsne.fit_transform(S_hat)

#%%
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

ax.scatter(
    X_hat[:cutoff, 0], X_hat[:cutoff, 1], c="C0", alpha=0.5, s=5, linewidth=0, label="V1"
)
ax.scatter(
    X_hat[cutoff:, 0], X_hat[cutoff:, 1], c="C1", alpha=0.5, s=5, linewidth=0, label="V2"
)
ax.set_aspect("equal")
ax.axis("off")
lgnd = plt.legend(fontsize=14)

for lg in lgnd.legendHandles:
    lg._sizes = [50]
    lg.set_alpha(1)
plt.show()
