# %%
# %cd ../
# %config InlineBackend.figure_format='retina'

import logging

import seaborn as sns
from src.receptive_field.rf import ReceptiveField, gen_rf_rank_regional
from src.spikeloader import SpikeLoader

sns.set()
logging.getLogger().setLevel(logging.INFO)

# %% tags=["parameters"]
path_loader = "data/superstim.hdf5"

# %% [markdown]
"""
# Ridge Regression for Receptive Field

$\underbrace{Y}_{t×n} = \underbrace{X}_{t×px} \underbrace{β}_{px×n}$
where
- $t$: number of time points
- $n$: number of neurons
- $px$: number of pixels in each image (stimulus)

Solve for $\hat{β}$ where $\hat{β} = \arg\min_β ||Y-Xβ||_2 + λ||β||_2$.
"""

# %%
loader = SpikeLoader.from_hdf5(path_loader)
rf = ReceptiveField(loader.img_dim, lamda=1.1)
rf.fit_neuron(loader.imgs_stim, loader.S)
rf.plot()
rf.save_append(path_loader, overwrite_group=True)

# %% [markdown]
"""
### Denoise

We use PCA to denoise the RFs. However, the location of the RFs vary across the visual cortex and linear models are not translation and rotation-invariant. Therefore, we split the visual cortex into blocks and perform PCA separately for each block.
"""
# %%
rf_pcaed = gen_rf_rank_regional(loader, rf, xy_div=(5, 3), plot=True)

# path_loader = Path(path_loader)
# np.save(path_loader.parent / (path_loader.stem + "rf_pcaed.npy"), rf_pcaed)
