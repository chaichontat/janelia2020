# %%
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

# %% [markdown]
#  ### Canonical Correlation Analysis
# 
#  Goal: compare the neural representations between V1 and V2
# 
#  Let $X$ and $Y$ be an $(n \times p)$ matrix where $n$ is the number of stimuli and $p$ is the number of neurons.
# 
#  We first split the spiking data of each region into two for a comparison between intra-region and inter-region CCA. Hence, we have 3 different CCA models to fit.
# 
#  | $X$  |  $Y$ |
#  |------|------|
#  | V1-1 | V1-2 |
#  | V1-1 | V2-1 |
#  | V2-1 | V2-2 |
# 
#  Then, for each group, we split the spiking data by stimulus into train and test sets.

# %%
loader = SpikeLoader.from_hdf5("data/superstim_TX57.hdf5")
gabor = GaborFit.from_hdf5("data/superstim_TX57_gabor.hdf5")

# %%
idx_spont = loader.idx_spont
spks = zscore(loader.spks, axis=0)
spks_nospont = SubtractSpontAnalyzer(128).fit(spks, loader.idx_spont).transform(spks)

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

    # df_all["group"] = 0
    # df_all.loc[df_all.sample(frac=0.5, replace=False).index, "group"] = 1
    # df_all["group"] = df_all["group"].astype("category")
    return df_all


regions = {"brain": (dict(group=0), dict(group=1))}

cr = CCARepeatedStim(loader, gabor, prepare_df=prepare_df, regions=regions)

# %%
ns_train = [20000]
rep, no_rep = loader.get_idx_rep(return_onetimers=True)
with cr.set_spks_source(spks_nospont[loader.istim.index, :]):
    df_un = cr.calc_cr(ns_train, idx_train=no_rep)

# %%
n_rep = loader.get_idx_rep().shape[0]
rand = np.random.default_rng(42)

with cr.set_spks_source(spks_nospont):
    innerprod_between_tests = [
        cr.calc_innerprod_test(
            df_un,
            idxs_test={
                "rep1": cr.loader.get_idx_rep(stim_idx=False)[:, 0],
                "rep2": cr.loader.get_idx_rep(stim_idx=False)[:, 1],
                "scrambled": rand.choice(np.arange(len(cr.S)), size=n_rep, replace=False),
                "spont": cr.loader.idx_spont[:n_rep],
            },
            pairs=[
                ("rep1", "rep1"),
                ("rep1", "rep2"),
                ("rep2", "rep2"),
                ("scrambled", "scrambled"),
                ("rep1", "scrambled"),
                ("spont", "spont"),
                ("rep1", "spont"),
            ],
            normalize=boo,
        )
        for boo in [True, False]
    ]

# %%
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

gen_chart(innerprod_between_tests[0]) | gen_chart(innerprod_between_tests[1])

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
