from typing import Any, Dict, List, Tuple, Union, Optional
import logging

import matplotlib.pyplot as plt
import numpy as np
from numpy.core.records import ndarray
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split

from src.canonical_analysis.canonical_ridge import CanonicalRidge
from src.gabor_analysis.gabor_fit import GaborFit
from src.spikeloader import SpikeLoader


class CCARegions:
    regions = {
        "V1V1": (dict(group=0, region="V1"), dict(group=1, region="V1")),
        "V1V2": (dict(group=0, region="V1"), dict(group=0, region="V2")),
        "V2V2": (dict(group=0, region="V2"), dict(group=1, region="V2")),
    }

    def __init__(
        self, loader: SpikeLoader, gabor: GaborFit, n_cc: int = 200, seed: int = 85
    ) -> None:
        self.loader = loader
        self.gabor = gabor
        self.seed = seed
        self.S = self.loader.S
        self.n_cc = n_cc

        self.df = self.prepare_df()

    @property
    def df_all(self) -> pd.DataFrame:
        d = pd.DataFrame(data=self.gabor.params_fit, columns=GaborFit.KEY.keys())
        d.rename(columns=dict(pos_x="azimuth", pos_y="altitude"), inplace=True)
        return d.join(self.loader.pos)

    def prepare_df(self, V2_size: float = 0.4, V2_cutoff: int = 180) -> pd.DataFrame:
        df_all = self.df_all.copy()
        # Line separating V1 and V2.
        df_all["region"] = pd.Categorical(np.where(df_all.y > V2_cutoff, "V1", "V2"))
        n_V2 = int(V2_size * df_all.groupby("region").size()["V2"])

        df_all["sampled"] = False
        df_all.loc[self._match_dist(df_all, "V1", "V2", size=n_V2), "sampled"] = True  # V2
        df_all.loc[df_all[df_all.region == "V1"].sample(n_V2).index, "sampled"] = True  # V1
        df_sampled = df_all[df_all.sampled]

        # Split each region into 2.
        df_sampled["group"] = 0
        df_sampled.loc[
            df_sampled.groupby("region")
            .apply(lambda s: s.sample(n_V2 // 2, random_state=self.seed))
            .index.get_level_values(1),
            "group",
        ] = 1
        df_sampled["group"] = df_sampled["group"].astype("category")
        return df_sampled

    def _match_dist(self, df, source, target, size):
        # Sample each region with matching distribution.
        assert source in df.region.cat.categories
        assert target in df.region.cat.categories

        azi = {reg: df[df.region == reg]["azimuth"] for reg in [source, target]}
        kde = {reg: gaussian_kde(z) for reg, z in azi.items()}
        p = (
            p_unnorm := kde[source].evaluate(azi[target]) / kde[target].evaluate(azi[target])
        ) / np.sum(p_unnorm)

        rand = np.random.default_rng(self.seed)
        return rand.choice(df[df.region == target].index, size=int(size), replace=False, p=p)

    @staticmethod
    def _gen_idxs_neuron(
        df: pd.DataFrame, region_pair: Tuple[Dict[str, Any], ...]
    ) -> List[pd.Index]:
        """
        Return the indices of neurons as filtered by the `pair` arg.

        Args:
            df (pd.DataFrame): [description]
            pair (Tuple[Dict[str, Any], ...]): A tuple of dicts that specifies filters {column: value}.
                Multiple filters are treated as intersections (AND).

        Returns:
            List[pd.Index]: List of neuron indices.
        """
        return [df.loc[df[p.keys()].isin(p.values()).all(axis=1), :].index for p in region_pair]

    def run_cca(
        self,
        idx_train: np.ndarray,
        idx_test: Optional[np.ndarray] = None,
        return_obj: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Run CCA with regions as specified in `self.regions`.
        Canonical vectors are generated from `self.S[idx_stim_train]`.
        Canonical coefs are from both `self.S[idx_stim_train]` and `self.S[idx_stim_test]`.

        Args:
            idx_stim_train (np.ndarray): [description]
            idx_stim_test (np.ndarray): [description]

        Returns:
            pd.DataFrame: canonical coefs in a tidy df. Columns are [coefs, regions, split].
        """

        out: List[pd.DataFrame] = list()
        cr_obj: List[CanonicalRidge] = list()

        for name, region_pair in self.regions.items():
            # Get neuron indices.
            idxs_neu = self._gen_idxs_neuron(self.df, region_pair)
            region1, region2 = [self.S[:, idx] for idx in idxs_neu]
            cr = CanonicalRidge(self.n_cc, lambda_x=0.85, lambda_y=0.85).fit(
                region1[idx_train], region2[idx_train]
            )
            out.append(pd.DataFrame(cr.coef).assign(regions=name, split="train"))
            if idx_test is not None:
                out.append(
                    pd.DataFrame(
                        cr.calc_canon_coef(region1[idx_test], region2[idx_test])
                    ).assign(regions=name, split='test')
                )

            if return_obj:
                cr_obj.append(pd.DataFrame(dict(cr=cr, regions=name), index=[0]))

        for d in out:
            d.reset_index(inplace=True)
            d.rename(columns={"index": "dimension", 0: "coef"}, inplace=True)

        if return_obj:
            return pd.concat(out), pd.concat(cr_obj, ignore_index=True)
        else:
            return pd.concat(out)

    def run_cca_transform(
        self, cr: CanonicalRidge, regions: str, idx_test: np.ndarray, split: str = "test"
    ) -> Tuple[np.ndarray, np.ndarray]:
        region_pair = self.regions[regions]
        idxs_neu = self._gen_idxs_neuron(self.df, region_pair)
        return cr.transform(*[self.S[idx_test][:, i] for i in idxs_neu])

    def run_random_splits(self, n_train: List[int], test_size=0.2) -> pd.DataFrame:
        res = dict()
        for n in n_train:
            idx_stim_tr, idx_stim_te = train_test_split(
                np.arange(0, int(n)), test_size=test_size, random_state=self.seed
            )
            res[n] = self.run_cca(idx_train=idx_stim_tr, idx_test=idx_stim_te).assign(n=n)

        return pd.concat(res.values())


class CCARepeatedStim(CCARegions):
    """
    All indices defined in this class are for `stim`.
    All neuron indices are already dealt with in `CCARegions`.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.idx_test = self.loader.get_idx_rep()[:, 1]
        self.idx_unrepeated = np.where(
            np.isin(range(self.S.shape[0]), self.loader.get_idx_rep().flatten(), invert=True)
        )[0]

    def _get_idx_train_repeated(self, n: int) -> pd.DataFrame:
        rand = np.random.default_rng(self.seed)
        idx_train_norep = pd.DataFrame(
            rand.choice(self.idx_unrepeated, size=n, replace=False)
        ).assign(test_stim_in_train=False)

        if n > len(self.idx_test):
            idx_train_rep = np.concatenate(
                (
                    self.loader.get_idx_rep()[:, 0],
                    rand.choice(
                        self.idx_unrepeated, size=n - len(self.idx_test), replace=False
                    ),
                )
            )
        else:
            idx_train_rep: np.ndarray = self.loader.get_idx_rep()[:n, 0]
        idx_train_rep = pd.DataFrame(idx_train_rep).assign(test_stim_in_train=True)

        return pd.concat([idx_train_norep, idx_train_rep]).rename(columns={0: "stim"})

    def run_repeated_splits(self, n_train: List[int]) -> pd.DataFrame:
        res = list()
        for n in n_train:
            idx = self._get_idx_train_repeated(n)
            for case in [True, False]:
                coefs = self.run_cca(
                    idx_train=idx[idx.test_stim_in_train.eq(case)]["stim"],
                    idx_test=self.idx_test,
                ).assign(test_stim_in_train=case, n=n)
                res.append(coefs)

        return pd.concat(res)

    def run_unrepeated(self, n_train: List[int]) -> pd.DataFrame:
        rand = np.random.default_rng(self.seed)
        res = list()
        idxs_repeated = {
            f"rep{i}": x.squeeze()
            for i, x in enumerate(np.split(self.loader.get_idx_rep(), 2, axis=1))
        }
        for n in n_train:
            _, objs = self.run_cca(
                idx_train=rand.choice(self.idx_unrepeated, size=n, replace=False),
                return_obj=True,
            )
            res.append(objs.assign(n=n))
        return pd.concat(res)

    def corr_between_test(self, df_cr: pd.DataFrame) -> pd.DataFrame:
        res = list()
        for row in df_cr.iloc:
            ted = [
                self.run_cca_transform(row.cr, regions=row.regions, idx_test=self.loader.get_idx_rep()[:, i])
                for i in range(2)
            ]
            oneone = np.array([np.corrcoef(ted[0][0][:, i], ted[0][1][:, i])[0, 1] for i in range(ted[0][0].shape[1])])
            onetwo = np.array([np.corrcoef(ted[0][0][:, i], ted[1][1][:, i])[0, 1] for i in range(ted[0][0].shape[1])])
            twotwo = np.array([np.corrcoef(ted[1][0][:, i], ted[1][1][:, i])[0, 1] for i in range(ted[0][0].shape[1])])
            res.append(pd.DataFrame(oneone, columns=['corr']).assign(match='rep1rep1', regions=row.regions, n=row.n))
            res.append(pd.DataFrame(onetwo, columns=['corr']).assign(match='rep1rep2', regions=row.regions, n=row.n))
            res.append(pd.DataFrame(twotwo, columns=['corr']).assign(match='rep2rep2', regions=row.regions, n=row.n))
        return pd.concat(res).reset_index().rename(columns={'index': 'dimension'})
            

    def calc_repeated_corr(self, n: int) -> pd.DataFrame:
        out = list()
        for name, pair in self.regions.items():  # Each region (V1V1, V1V2, ...).
            # Get neuron indices.
            idxs_neu = self._gen_idxs_neuron(self.df, pair)

            for group, idx_neu in enumerate(idxs_neu):  # Each group in pair (V1-1, ...).
                S_neu_filtered = pd.DataFrame(self.S[self.idx_test][:, idx_neu])

                for case in [True, False]:
                    idx = self._get_idx_train_repeated(n)
                    df_train = pd.DataFrame(
                        self.S[idx[idx.test_stim_in_train.eq(case)]["stim"]][:, idx_neu]
                    )
                    corr = pd.DataFrame(df_train.corrwith(S_neu_filtered)).assign(
                        regions=name, group=group, test_stim_in_train=case, n=n
                    )
                    out.append(corr)

        return pd.concat(out).rename(columns={0: "corr"})


if __name__ == "__main__":
    from scripts.subspace_comm import CCARepeatedStim
    from src.gabor_analysis.gabor_fit import GaborFit
    from src.spikeloader import SpikeLoader

    cr = CCARepeatedStim(
    SpikeLoader.from_hdf5("data/processed.hdf5"), GaborFit.from_hdf5("data/gabor.hdf5")
    )
    n_train = [500, 1000, 2000, 5000, 10000, 20000]
    df_un = cr.run_unrepeated(n_train=n_train[:3])
    test = cr.corr_between_test(df_un)
    # %% [markdown]
    # V1 and V2 are separated by a line where the azimuth preference reverses with increased receptive field size (σ).
    #
    # The retinotopy between both regions are matched by a method similar to importance sampling.

    # %%

    # %% [markdown]
    # There is a sharp increase in sampled neuron at the V1-V2 boundary. This is due to the fact that V1 neurons outnumber V2 neurons by 60%. Furthermore, the azimuthal preferences of V2 neurons extend include more of the lateral visual field, reducing the number of potential matches with V1.

    # %% [markdown]
    # We randomly split each region into two for CCA and verify that the split is uniform.

    # %%

    # %% [markdown]
    # We perform CCA with an 80:20 train:test stimuli split.

    # %%

    # from statsmodels.nonparametric.smoothers_lowess import lowess
    # dfs = []
    # for sens in df_coefs.variable.unique():
    #     for meas in df_coefs.n.unique():
    #         # One independent smoothing per Sensor/Measure condition.
    #         df_filt = df_coefs.loc[(df_coefs.variable == sens) & (df_coefs.n == meas)]
    #         # Frac is equivalent to span in R
    #         filtered = lowess(df_filt['canonical coef'], df_filt.dimension, frac=0.05)
    #         df_filt["filteredvalue"] = filtered[:,1]
    #         dfs.append(df_filt)
    # df_lowess = pd.concat(dfs)

    # %% [markdown]
    # ### Repeated Stimuli
    #
    # The test set now contains responses from the second repeat of the repeated stimuli. Here, we test whether the presence of the responses to an identical set of stimuli in the training set would affect the results.

    # %%

    # %%

    # %% [markdown]
    # ### Verify Train/Test Split
    #
    # First, we check if the given indices are correct.

    # %%
    # rep = [pd.DataFrame(f.S[f.get_idx_rep()[:, i], :]) for i in range(2)]
    # ctrl = rep[0].corrwith(pd.DataFrame(f.S[:2141, :]))  # First 2141 stim
    # test = rep[0].corrwith(rep[1])

    # # %%
    # sns.distplot(ctrl, label="Rep1+Ctrl")
    # ax = sns.distplot(test, label="Rep1+Rep2")
    # ax.set_xlabel("Correlation")
    # ax.set_title("Correlation, all repeated stim, all neurons")
    # plt.legend()

    # %% [markdown]
    # Next, we extract the data matrix generated above and calculate the correlations between train and test sets for all groups/regions.

