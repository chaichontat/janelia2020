# %%
from IPython import get_ipython

get_ipython().run_line_magic("cd", "../")

import altair as alt
import seaborn as sns
from src.gabor_analysis.gabor_fit import GaborFit
from src.receptive_field.rf import ReceptiveField, gen_rf_rank_regional
from src.spikeloader import SpikeLoader
from src.utils.plots import gabor_interactive

get_ipython().run_line_magic("matplotlib", "inline")
get_ipython().run_line_magic("config", "InlineBackend.figure_format='retina'")
sns.set()

# %%
f = SpikeLoader.from_hdf5("data/superstim_TX60_allsort.hdf5")
rf = ReceptiveField.from_hdf5("data/superstim_TX60_allsort.hdf5")
g = GaborFit.from_hdf5("data/superstim_TX60_allsort_gabor.hdf5")

# %%
rf_pcaed = gen_rf_rank_regional(f, rf, xy_div=(5, 3))
# %%
g.plot_corr(rf.rf_, rf_pcaed)

# %%
g.plot_params(f.pos)

# %% [markdown]
# ## Interactive Plot

# %%
alt.renderers.enable('mimetype')
gabor_interactive(f, g, n_samples=500)
