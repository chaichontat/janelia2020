import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from .canonical_ridge import CanonicalRidge
from ..receptive_field.rf import ReceptiveField
from ..spikeloader import SpikeLoader

sns.set()
loader = SpikeLoader()
cca: CanonicalRidge = pickle.loads(Path('cc.pk').read_bytes())

#%% Plot PCs after CCs subtraction.
V1s, V2s = cca.subtract_canon_comp(cca.X_ref, cca.Y_ref)
rf_v1 = ReceptiveField(loader.img_dim).fit_neuron(loader.imgs_stim, V1s)
rf_v1.plot()
rf_v2 = ReceptiveField(loader.img_dim).fit_neuron(loader.imgs_stim, V2s)
rf_v2.plot()

plt.show()
