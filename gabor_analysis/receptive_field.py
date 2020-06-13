#%%
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import seaborn as sns

from scipy.ndimage import gaussian_filter
from scipy.stats import zscore
from sklearn.decomposition import PCA

sns.set()

#%%
def make_receptive(S, img, λ=1., pca=True):
    # Process spikes.
    NN, NT = S.shape
    S = zscore(S)
    if pca:
        pca_model = PCA(n_components=n_pca)
        pca_model.fit(S)
        Sp = zscore(pca_model.components_, axis=1)
    else:
        Sp = S

    # Process stimuli.
    ly, lx, nstim = img.shape
    X = np.reshape(img, [-1, NT])  # (pixels x t)
    X = zscore(X, axis=1) / np.sqrt(NT)  # z-score each pixel separately

    # Linear regression
    B0 = np.linalg.solve((X @ X.T + λ * np.eye(X.shape[0])), X @ Sp.T)
    B0 = np.reshape(B0, [ly, lx, -1])
    B0 = gaussian_filter(B0, [.5, .5, 0])

    return B0.transpose((2, 0, 1))

def gen_plot(B0):
    rfmax = np.max(B0)
    fig, axs = plt.subplots(nrows=5, ncols=4, figsize=(10, 8), dpi=300)
    axs = axs.flatten()
    for i in range(20):
        axs[i].imshow(B0[:, :, i], cmap='bwr', vmin=-rfmax, vmax=rfmax)
        axs[i].axis('off')
        axs[i].set_title(f'PC {i}')

    plt.show()


#%% Pre-processing
print('Loading data.')
raw = np.load('/Users/chaichontat/Downloads/superstim.npz')
for name in raw.files:
    globals()[name] = raw[name]

scale = 0.25
n_pca = 100

img = ndi.zoom(img, (scale, scale, 1), order=1)  # Resize

#%%
B = make_receptive(spks[:, frame_start], img[:, :, istim], pca=False)

with open('field.pk', 'wb') as f:
        pickle.dump(B, f)

#%% V1 and V2
# thr = 250
# spks = spks[:, frame_start]
# V = spks[ypos < thr, :]

# for v in [spks[ypos < thr, :], spks[ypos >= thr, :]]:
#     B = make_receptive(v)
#     gen_plot(B)

#%% Train/Test
np.random.seed(342)
fold = 2
idx = np.random.choice(np.arange(len(frame_start)), len(frame_start) // fold)

B = make_receptive(spks[:, frame_start[idx]], img[:, :, istim[idx]], pca=False)
with open('field1.pk', 'wb') as f:
    pickle.dump(B, f)


# Inverse
mask = np.ones(len(frame_start), dtype=np.bool)
mask[idx] = 0
idx = np.arange(len(frame_start))[mask]

B = make_receptive(spks[:, frame_start[idx]], img[:, :, istim[idx]], pca=False)
with open('field2.pk', 'wb') as f:
    pickle.dump(B, f)
