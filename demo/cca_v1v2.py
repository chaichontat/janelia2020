# -*- coding: utf-8 -*-
# %%
# %cd ../

from IPython import get_ipython
get_ipython().run_line_magic("config", "InlineBackend.figure_format='retina'")

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.canonical_analysis.subspace_comm import CCARepeatedStim
from src.gabor_analysis.gabor_fit import GaborFit
from src.power_law.subtract_spont import SubtractSpontAnalyzer
from src.spikeloader import SpikeLoader

alt.data_transformers.disable_max_rows()
sns.set()

# %% [markdown]
#  ### Canonical Correlation Analysis
#
#  Goal: compare the neural representations between V1 and V2
#
#  Let $X$ and $Y$ be an $(n \times p)$ matrix where $n$ is the number of stimuli and $p$ is the number of neurons.
#
#  We first split the spiking data of each region into two for a comparison between intra-region and inter-region CCA. Hence, we have 3 different CCA models to fit.
#
#  | $X$ | $Y$ |
#  |------|------|
#  | V1-1 | V1-2 |
#  | V1-1 | V2-1 |
#  | V2-1 | V2-2 |
#
#  Then, for each group, we split the spiking data by stimulus into train and test sets.

# %%
path_loader = "data/superstim.hdf5"
path_gabor = "data/superstim.hdf5"

# %%
cr = CCARepeatedStim(
    loader := SpikeLoader.from_hdf5(path_loader),
    gabor := GaborFit.from_hdf5(path_gabor),
)
n_train = [500, 1000, 2000, 5000, 10000]

# %%
sns.pairplot(
    cr.df,
    hue="region",
    vars=["x", "y", "σ", "azimuth", "altitude"],
    corner=True,
    plot_kws=dict(s=4, linewidth=0, alpha=0.3),
)

# %% [markdown]
# There is a sharp increase in sampled neuron at the V1-V2 boundary. This is due to the fact that V1 neurons outnumber V2 neurons by 60%. Furthermore, the azimuthal preferences of V2 neurons extend include more of the lateral visual field, reducing the number of potential matches with V1.
#
# We perform CCA with an 80:20 train:test stimuli split with various numbers of training stimuli.

# %%
def gen_chart(data: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(data[data.split == "test"])
        .mark_line(opacity=0.5)
        .encode(
            x=alt.X("dimension:Q"),  # , scale=alt.Scale(domain=xlim)),
            y=alt.Y("coef:Q"),  # , scale=alt.Scale(domain=ylim)),
            color="n:N",
            column=alt.Column("regions:N"),
        )
    ).properties(width=200, height=300)

df_rand = cr.run_random_splits(ns=n_train)
gen_chart(df_rand).properties(title=("Inter- and Intra-region Canonical Coefficients"))

# %% [markdown]
# The test set now contains responses from the second repeat of the repeated stimuli. Here, we test whether the presence of the responses to an identical set of stimuli in the training set would affect the results.

# %%
df_repeated = cr.run_repeated_splits(ns_train=n_train)
gen_chart(df_repeated[df_repeated.split == "test"]).encode(
    alt.Row("test_stim_in_train")
).properties(title=("Canonical Coefficients: Repeated Stimuli"), height=200)

# %% [markdown]
# ### Verify Train/Test Split
# First, we check if the indices are correct.

# %%
df_corr = cr.calc_repeated_corr(n=2000)
sns.FacetGrid(df_corr, col="regions", row="group", hue="test_stim_in_train").map(
    plt.hist, "corr", alpha=0.6
)

# %% [markdown]
# How does training on an unrelated stimuli affect things?

# %%
rep, no_rep = loader.get_idx_rep(return_onetimers=True)
df_un = cr.calc_cr(ns_train=[10000], idx_train=no_rep)
corr_between_tests = cr.calc_innerprod_test(
    df_un,
    idxs_test={
        "rep1": cr.loader.get_idx_rep(stim_idx=False)[:, 0],
        "rep2": cr.loader.get_idx_rep(stim_idx=False)[:, 1],
        "ctrl": cr.loader.istim.iloc[range(2141)],
        "spont": cr.loader.idx_spont[:2141],
    },
    pairs=[
        ("rep1", "rep1"),
        ("rep1", "rep2"),
        ("ctrl", "ctrl"),
        ("rep1", "ctrl"),
        ("spont", "spont"),
        ("rep1", "spont"),
    ],
    stim_idx=False,
    normalize=True
)

# %% [markdown]
# Here, we generate canonical vectors from all unrepeated stimuli. These are then used to separately generate canonical variates for the repeated stimuli. We then calculate the correlation (canonical coefficient) within and between repeats.

# %%
corr_between_tests['compare_type'] = np.where(corr_between_tests['match'].isin(['ctrl_ctrl', 'rep1_rep1', 'spont_spont']), 'same', 'different')

selection = alt.selection_multi(fields=["match"], bind="legend")

base = alt.Chart(corr_between_tests[corr_between_tests.n > 2000]).encode(
    x="dimension", y="corr", color="match",
)

base.mark_line().encode(
    strokeDash=alt.StrokeDash("compare_type", sort="descending"),
    size=alt.condition(~selection, alt.value(1), alt.value(2)),
    opacity=alt.condition(~selection, alt.value(0.4), alt.value(1)),
    column="regions",
    row="n:N",
).properties(width=200, height=250).add_selection(selection)

# %% [markdown]
# The same analysis, with spontaneous activities subtracted.

# %%
spks_nospont = SubtractSpontAnalyzer(128).fit(loader.spks, loader.idx_spont).transform(loader.spks)
with cr.set_spks_source(spks_nospont[loader.istim.index, :]):
    df_un = cr.calc_cr([5000, 10000], no_rep)

# %%
with cr.set_spks_source(spks_nospont):
    corr_between_tests = cr.calc_innerprod_test(
        df_un,
        idxs_test={
            "rep1": cr.loader.get_idx_rep(stim_idx=False)[:, 0],
            "rep2": cr.loader.get_idx_rep(stim_idx=False)[:, 1],
            "ctrl": cr.loader.istim.iloc[range(2141)],
            "spont": cr.loader.idx_spont[:2141],
        },
        pairs=[
            ("rep1", "rep1"),
            ("rep1", "rep2"),
            ("ctrl", "ctrl"),
            ("rep1", "ctrl"),
            ("spont", "spont"),
            ("rep1", "spont"),
        ],
    )


# %%
corr_between_tests['compare_type'] = np.where(corr_between_tests['match'].isin(['ctrl_ctrl', 'rep1_rep1', 'spont_spont']), 'same', 'different')

selection = alt.selection_multi(fields=["match"], bind="legend")

base = alt.Chart(corr_between_tests[corr_between_tests.n > 2000]).encode(
    x="dimension", y="corr", color="match",
)

base.mark_line().encode(
    strokeDash=alt.StrokeDash("compare_type", sort="descending"),
    size=alt.condition(~selection, alt.value(1), alt.value(2)),
    opacity=alt.condition(~selection, alt.value(0.4), alt.value(1)),
    column="regions",
    row="n:N",
).properties(width=200, height=250).add_selection(selection)
