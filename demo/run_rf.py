# %%
# %cd ../

from IPython.core.getipython import get_ipython

import logging

import numpy as np
import seaborn as sns
from src.receptive_field.rf import ReceptiveField, gen_rf_rank_regional
from src.spikeloader import SpikeLoader

sns.set()
logging.getLogger().setLevel(logging.INFO)

# %%
path_loader = "data/superstim.hdf5"

# %%
loader = SpikeLoader.from_hdf5(path)
rf = ReceptiveField(loader.img_dim, lamda=1.1)
rf.fit_neuron(loader.imgs_stim, loader.S)
rf.plot()
rf.save_append(path)

# %%
rf_pcaed = gen_rf_rank_regional(loader, rf, xy_div=(5, 3), plot=True)
np.save("data/rf_pcaed.npy", rf_pcaed)
