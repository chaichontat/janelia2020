#%%
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import seaborn as sns
from scipy.ndimage import gaussian_filter
from scipy.stats import zscore
from sklearn.decomposition import PCA


def make_receptive(S: np.ndarray, img: np.ndarray,
                   λ: float = 1., n_pcs: int = 100) -> np.ndarray:
    """
    Generate receptive fields (RFs) by multivariate linear regression between spikes and pixels.
    Can generate RFs of each principal component (PC) OR each neuron.

    Parameters
    ----------
    S: np.ndarray
        Matrix of spikes for each stimulus (n_neu x n_stim).
    img: np.ndarray
        Matrix of stimulus images (y x x x n_stim)
    λ: float
        Coefficient for L2-regularization of the least-squares regression.
    n_pcs: int
        Number of PCs to generate RFs of. OR
        0 -> generate RF of each neuron.

    Returns
    -------
    B0: np.ndarray
        RFs in (y x x x `pca`) OR (y x x x n_stim).

    """
    NN, NT = S.shape
    ly, lx, nstim = img.shape
    assert NT == nstim

    # Process spikes.
    S = zscore(S, axis=1)  # Individual neuron.
    if n_pcs:
        print('PCAing.')
        pca_model: PCA = PCA(n_components=n_pcs).fit(S)
        Sp = zscore(pca_model.components_, axis=1)
    else:
        Sp = S

    # Process stimuli.
    X = np.transpose(img, (2, 0, 1))
    X = np.reshape(X, [NT, -1])  # (stim x pxs)
    X = zscore(X, axis=0)  # z-score each pixel separately

    # Linear regression
    print('Running linear regression.')
    B0 = np.linalg.solve((X.T @ X + λ * np.eye(X.shape[1])), X.T @ Sp.T)
    B0 = np.reshape(B0, [ly, lx, -1])
    B0 = gaussian_filter(B0, [.5, .5, 0])
    return B0


def gen_rf_plot(B0: np.ndarray) -> None:
    """
    Generate a grid of RFs.

    Parameters
    ----------
    B0: np.ndarray
        RFs in (y x x x n_stim).

    Returns
    -------
    None
    """

    rfmax = np.max(np.abs(B0))
    fig, axs = plt.subplots(nrows=5, ncols=4, figsize=(10, 8), dpi=300)
    axs = axs.flatten()
    for i in range(20):
        axs[i].imshow(B0[:, :, i], cmap='bwr', vmin=-rfmax, vmax=rfmax)
        axs[i].axis('off')
        axs[i].set_title(f'PC {i + 1}')
    plt.show()


if __name__ == '__main__':
    sns.set()

    # %% Pre-processing
    print('Loading data.')
    with np.load('superstim.npz') as npz:
        img, spks = npz['img'], npz['spks']
        xpos, ypos = npz['xpos'], npz['ypos']
        frame_start, istim = npz['frame_start'], npz['istim']

    # Resize image.
    scale = 4.
    img = ndi.zoom(img, (1 / scale, 1 / scale, 1), order=1)

    S = spks[:, frame_start]
    X = img[:, :, istim]

    # %% Generate RFs for PCs.
    B = make_receptive(S, X, n_pcs=20)
    gen_rf_plot(B)

    #%% Generate RFs for every neuron.
    B = make_receptive(S, X, n_pcs=0)
    with open('field.pk', 'wb') as f:
        pickle.dump(B, f)

    #%% Split data randomly into two and fit RFs on each.
    np.random.seed(342)
    idx = np.random.choice(np.arange(len(frame_start)), len(frame_start) // 2)

    # Invert indices.
    mask = np.ones_like(frame_start, dtype=np.bool)
    mask[idx] = 0
    idx_inv = np.arange(len(frame_start))[mask]

    for i, u in enumerate([idx, idx_inv]):
        B = make_receptive(spks[:, frame_start[u]], img[:, :, istim[u]], n_pcs=0)
        with open(f'field{i+1}.pk', 'wb') as f:
            pickle.dump(B, f)
