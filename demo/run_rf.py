# %%
# %cd ../
import logging
from pathlib import Path

import numpy as np
import seaborn as sns
from src.receptive_field.rf import ReceptiveField, gen_rf_rank_regional
from src.spikeloader import SpikeLoader

sns.set()
logging.getLogger().setLevel(logging.INFO)

# %% tags=["parameters"]
path_loader = "data/superstim.hdf5"

# %%
loader = SpikeLoader.from_hdf5(path_loader)
rf = ReceptiveField(loader.img_dim, lamda=1.1)
rf.fit_neuron(loader.imgs_stim, loader.S)
rf.plot()
rf.save_append(path_loader, overwrite_group=True)

# %%
rf_pcaed = gen_rf_rank_regional(loader, rf, xy_div=(5, 3), plot=True)

# path_loader = Path(path_loader)
# np.save(path_loader.parent / (path_loader.stem + "rf_pcaed.npy"), rf_pcaed)
