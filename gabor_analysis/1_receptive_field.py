#%%
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import seaborn as sns

from scipy.ndimage import gaussian_filter
from scipy.stats import zscore
from sklearn.decomposition import PCA


def make_receptive(S, img, λ=1., pca=100):
    NN, NT = S.shape
    ly, lx, nstim = img.shape
    assert NT == nstim

    # Process spikes.
    S = zscore(S, axis=1)
    if pca:
        print('PCAing.')
        pca_model = PCA(n_components=pca)
        pca_model.fit(S)
        Sp = zscore(pca_model.components_, axis=1)
    else:
        Sp = S

    # Process stimuli.
    X = np.transpose(img, (2, 0, 1))
    X = np.reshape(X, [NT, -1])  # time x pxs
    X = zscore(X, axis=0)  # z-score each pixel separately

    # Linear regression
    print('Running linear regression.')
    B0 = np.linalg.solve((X.T @ X + λ * np.eye(X.shape[1])), X.T @ Sp.T)
    B0 = np.reshape(B0, [ly, lx, -1])
    B0 = gaussian_filter(B0, [.5, .5, 0])
    return B0


def gen_rf_plot(B0):
    rfmax = np.max(np.abs(B0))
    fig, axs = plt.subplots(nrows=5, ncols=4, figsize=(10, 8), dpi=300)
    axs = axs.flatten()
    for i in range(20):
        axs[i].imshow(B0[:, :, i], cmap='bwr', vmin=-rfmax, vmax=rfmax)
        axs[i].axis('off')
        axs[i].set_title(f'PC {i+1}')
    plt.show()


if __name__ == '__main__':
    sns.set()

    #%% Pre-processing
    print('Loading data.')
    raw = np.load('superstim.npz')
    for name in raw.files:
        globals()[name] = raw[name]

    # Resize image.
    scale = 4.
    img = ndi.zoom(img, (1/scale, 1/scale, 1), order=1)

    S = spks[:, frame_start]
    X = img[:, :, istim]


    #%% Generate RFs for PCs.
    B = make_receptive(S, X, pca=20)
    gen_rf_plot(B)


    #%% Generate RFs for every neuron.
    B = make_receptive(S, X, pca=False)
    with open('field.pk', 'wb') as f:
        pickle.dump(B, f)


    #%% Train/Test
    np.random.seed(342)
    idx = np.random.choice(np.arange(len(frame_start)), len(frame_start) // 2)

    # Inverse
    mask = np.ones_like(frame_start, dtype=np.bool)
    mask[idx] = 0
    idx_inv = np.arange(len(frame_start))[mask]

    for i, u in enumerate([idx, idx_inv]):
        B = make_receptive(spks[:, frame_start[u]], img[:, :, istim[u]], pca=False)
        with open(f'field{i+1}.pk', 'wb') as f:
            pickle.dump(B, f)
