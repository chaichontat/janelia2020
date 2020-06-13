import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

sns.set()

def pca(B, n_pca):
    model = PCA(n_components=n_pca)
    B_flatten = np.reshape(B, (len(B), -1))  # n_neu x pixels.
    B_reduced = (model.fit_transform(B_flatten) @ model.components_).reshape(B.shape)
    print(np.sum(model.explained_variance_ratio_))
    pcs = model.components_.reshape((-1, B.shape[1], B.shape[2]))

    return B_reduced, pcs


if __name__ == '__main__':
    n_pca = 90
    B = pickle.loads(Path('field.pk').read_bytes())
    B_reduced, pcs = pca(B, n_pca)

    #%% Sanity check.
    np.random.seed(535)
    idx = np.random.randint(low=0, high=B.shape[2], size=10)
    fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(10, 6), dpi=300)

    axs = np.hstack([axs[0:2, :], axs[2:4, :]]).T

    for i in range(10):
        scale = np.max(np.abs([B[:, :, idx[i]], B_reduced[:, :, idx[i]]]))
        axs[i, 0].imshow(B[:, :, idx[i]], cmap='bwr', vmin=-scale, vmax=scale)
        axs[i, 1].imshow(B_reduced[:, :, idx[i]], cmap='bwr', vmin=-scale, vmax=scale)
        axs[i, 0].axis('off')
        axs[i, 1].axis('off')
    fig.suptitle(f'10 randomly chosen receptive fields. \n L: Original, R: First {n_pca} PCs')
    plt.tight_layout()
    plt.show()
