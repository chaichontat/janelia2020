# %%
# %cd ../
import altair as alt
import seaborn as sns
from IPython import get_ipython
from src.gabor_analysis.gabor_fit import GaborFit
from src.receptive_field.rf import ReceptiveField, gen_rf_rank_regional
from src.spikeloader import SpikeLoader
from src.utils.plots import gabor_interactive

get_ipython().run_line_magic("matplotlib", "inline")
get_ipython().run_line_magic("config", "InlineBackend.figure_format='retina'")
sns.set()

# %%
path_loader = "data/superstim_TX60_allsort.hdf5"
path_rf = "data/superstim_TX60_allsort.hdf5"
path_gabor = "data/superstim_TX60_allsort_gabor.hdf5"

# %%
f = SpikeLoader.from_hdf5(path_loader)
rf = ReceptiveField.from_hdf5(path_rf)
g = GaborFit.from_hdf5(path_gabor)

# %%
g.plot_params(f.pos)

# %% [markdown]
# ## Interactive Plot

# %%
alt.renderers.enable("mimetype")
gabor_interactive(f, g, n_samples=500)
