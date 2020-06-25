import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from gabor_analysis.receptive_field import reduce_rf_rank

sns.set()

""" Denoise RFs by PCA. """

n_pca = 60
B = pickle.loads(Path('gabor_analysis/field.pk').read_bytes())
B_reduced, pcs = reduce_rf_rank(B, n_pca)

# %% Sanity check.
np.random.seed(535)
idx = np.random.randint(low=0, high=B.shape[2], size=10)
fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(10, 6), dpi=300, constrained_layout=True)

axs = np.hstack([axs[0:2, :], axs[2:4, :]]).T
for i in range(10):
    scale = np.max(np.abs([B[idx[i], :, :], B_reduced[idx[i], :, :]]))
    axs[i, 0].imshow(B[idx[i], :, :], cmap='bwr', vmin=-scale, vmax=scale)
    axs[i, 1].imshow(B_reduced[idx[i], :, :], cmap='bwr', vmin=-scale, vmax=scale)
    axs[i, 0].axis('off')
    axs[i, 1].axis('off')

fig.suptitle(f'10 randomly chosen receptive fields. \n Top: Original, Bottom: First {n_pca} PCs')
plt.show()
