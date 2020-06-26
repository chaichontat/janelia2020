import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import zscore
from sklearn.metrics import mean_squared_error

from .rf import ReceptiveField
from ..spikeloader import SpikeLoader

sns.set()

""" Denoise RFs by PCA. """

n_pc = 60
rf: ReceptiveField = pickle.loads(Path('field.pk').read_bytes())
B = rf.rf_
B_reduced = rf.gen_rf_rank(n_pc)

# %% Sanity check.
np.random.seed(535)
idx = np.random.randint(low=0, high=B_reduced.shape[2], size=10)
fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(10, 6), dpi=300, constrained_layout=True)

axs = np.hstack([axs[0:2, :], axs[2:4, :]]).T
for i in range(10):
    scale = np.max(np.abs([B[idx[i], :, :], B_reduced[idx[i], :, :]]))
    axs[i, 0].imshow(B[idx[i], :, :], cmap='bwr', vmin=-scale, vmax=scale)
    axs[i, 1].imshow(B_reduced[idx[i], :, :], cmap='bwr', vmin=-scale, vmax=scale)
    axs[i, 0].axis('off')
    axs[i, 1].axis('off')

fig.suptitle(f'10 randomly chosen receptive fields. \n Top: Original, Bottom: First {n_pc} PCs')
plt.show()

# %% CV
loader = SpikeLoader()
trX, teX, trS, teS = loader.train_test_split()
rf_1 = ReceptiveField(loader, λ=1.1).fit_neuron(trX, trS)
rf_2 = ReceptiveField(loader, λ=1.1).fit_neuron(teX, teS)


# %%
def norm(img):
    return zscore(img.reshape([img.shape[0], -1]), axis=1)


def objective(n):
    print(n)
    mse = mean_squared_error(norm(rf_1.gen_rf_rank(n)), norm(rf_2.rf_)) + \
          mean_squared_error(norm(rf_1.rf_), norm(rf_2.gen_rf_rank(n)))
    return float(mse)


ns = [4, 8, 10, 20, 30, 40, 50, 60, 80, 100]
results = [objective(n) for n in ns]
plt.plot(ns, results)
plt.show()

# Optimum is 30.
