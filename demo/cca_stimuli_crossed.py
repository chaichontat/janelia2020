# -*- coding: utf-8 -*-
# %%
# %cd ../
# %config InlineBackend.figure_format='retina'

import altair as alt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats.stats import zscore
from src.canonical_analysis.subspace_comm import CCARepeatedStim
from src.gabor_analysis.gabor_fit import GaborFit
from src.power_law.subtract_spont import SubtractSpontAnalyzer
from src.spikeloader import SpikeLoader

alt.data_transformers.disable_max_rows()
sns.set()

# %% tags=["parameters"]
path_loader = "data/superstim_TX60_allsort.hdf5"
path_gabor = "data/superstim_TX60_allsort_gabor.hdf5"

# %%
loader = SpikeLoader.from_hdf5(path_loader)
gabor = GaborFit.from_hdf5(path_gabor)

idx_spont = loader.idx_spont
spks = zscore(loader.spks, axis=0)
S_nospont = SubtractSpontAnalyzer(128).fit(spks, loader.idx_spont).transform(loader.S)

# %%
def prepare_df(df_all: pd.DataFrame):
    df_all = df_all.copy()
    df_all["region"] = "brain"

    def checkerboard(item: pd.Series):
        if ((item.x // 100) + (item.y // 100)) % 2 == 0:
            return 0
        else:
            return 1

    df_all["group"] = df_all.apply(checkerboard, axis=1).astype("category")
    return df_all


regions = {"brain": (dict(group=0), dict(group=1))}

cr = CCARepeatedStim(loader, gabor, prepare_df=prepare_df, regions=regions)

# %%
rep = loader.get_idx_rep()
n_rep = rep.shape[0]
rand = np.random.default_rng(42)
# tr, te = train_test_split(np.arange(n_rep), train_size=0.8, random_state=42)
sep = int(0.8 * n_rep)

ns_train = [sep]
with cr.set_spks_source(S_nospont):
    df_classic = cr.calc_cr(ns_train, idx_train=rep[:sep, 0], idx_train2=rep[:sep, 0])
    df_swap = cr.calc_cr(ns_train, idx_train=rep[:sep, 0], idx_train2=rep[:sep, 1])

# %%
idx_scrambled = rep[rand.integers(low=0, high=n_rep, size=n_rep - sep), 0]


def corr(df):
    with cr.set_spks_source(S_nospont):
        return cr.calc_innerprod_test(
            df,
            idxs_test={
                "rep1": rep[sep:, 0],
                "rep2": rep[sep:, 1],
                "scrambled": idx_scrambled,
                "training_rep1": rep[: n_rep - sep, 0],
                "training_rep2": rep[: n_rep - sep, 1],
            },
            pairs=[
                ("rep1", "rep1"),
                ("rep2", "rep2"),
                ("rep2", "rep1"),
                ("rep1", "rep2"),
                ("rep1", "scrambled"),
                ("training_rep1", "training_rep2"),
                ("training_rep1", "training_rep1"),
            ],
            normalize=True,
        )


def gen_chart(data: pd.DataFrame) -> alt.Chart:
    y = "cov" if "cov" in data.columns else "corr"
    selection = alt.selection_multi(fields=["match"], bind="legend")
    base = alt.Chart(data).encode(x="dimension", y=y, color="match",)

    return (
        base.mark_line()
        .encode(
            size=alt.condition(~selection, alt.value(1), alt.value(2)),
            opacity=alt.condition(~selection, alt.value(0.4), alt.value(1)),
            row="n:N",
        )
        .properties(width=200, height=250)
        .add_selection(selection)
    )


corr_classic, corr_swap = corr(df_classic), corr(df_swap)

alt.hconcat(
    gen_chart(corr_classic).properties(title="Classic"),
    gen_chart(corr_swap).properties(title="Swapped"),
)

# %% [markdown]
# ### Summary
# This is a plot of the correlation of canonical variates or projections
# between the activities of neuron groups 1 and 2, with different lines
# signifying different first/second repeat combinations.
#
# The name of each line is split by \_ the first is always neuron group1
# and the second is always neuron group2
#
# - rep2_rep1: rep2 for neuron group 1 and rep1 for neuron group 2.
#
# The scrambled is just some random index from group1 with some random index from group2.
# We’re pairing unrelated images and we expect a correlation of 0.
# The training is using the training dataset to establish an upper bound.
#
#
# ### Training
# 80:20 train/test split without randomization.
# - Classic: Group 1 and 2: stims from repeat 1.
# - Swapped: Group 1: stims from repeat 1. Group 2: stims from repeat 2.
#

# %%

# sns.FacetGrid(
#     data=df_transformed[df_transformed["stim"].isin(rand.integers(low=0, high=n_rep, size=15))],
#     col="stim",
#     col_wrap=5,
# ).map(
#     sns.regplot, "rep1", "rep2", scatter_kws={"s": 1, "alpha": 0.5},
# )
# sns.regplot(
#     "rep1",
#     "rep2",
#     data=df_transformed[df_transformed.stim == 0],
#     ax=ax,
#     scatter_kws={"s": 1, "alpha": 0.5},
# )
# ax.set_aspect("equal")
