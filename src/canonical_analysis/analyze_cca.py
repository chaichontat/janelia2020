import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from .canonical_ridge import CanonicalRidge
from ..gabor_analysis.receptive_field import ReceptiveField
from ..spikeloader import SpikeStimLoader

# %%
sns.set()
loader = SpikeStimLoader()
cca: CanonicalRidge = pickle.loads(Path('cc.pk').read_bytes())

V1s, V2s = cca.subtract_canon_comp(cca.X_ref, cca.Y_ref)

rf_v1 = ReceptiveField(loader).fit(loader.X, V1s)
rf_v1.plot_rf()

rf_v2 = ReceptiveField(loader).fit(loader.X, V2s)
rf_v2.plot_rf()

plt.show()
