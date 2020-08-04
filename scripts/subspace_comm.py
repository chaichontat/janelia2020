from contextlib import contextmanager
from typing import Any, Dict, List, Tuple, Union, Optional, overload, Literal

import logging

import matplotlib.pyplot as plt
import numpy as np
from numpy.core.records import ndarray
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde, zscore
from sklearn.model_selection import train_test_split

from src.canonical_analysis.canonical_ridge import CanonicalRidge
from src.gabor_analysis.gabor_fit import GaborFit
from src.spikeloader import SpikeLoader, LazyProperty


class CCARegions:
    """Class to perform CCA between brain regions (V1 and V2).
    Primarily to split neurons into regions and groups.
    
    See `subspace_nb.ipynb` for examples.
    """

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
        self.spks_source = None

    @LazyProperty
    def spks(self) -> np.ndarray:
        return zscore(self.loader.spks, axis=0)

    @LazyProperty
    def df_all(self) -> pd.DataFrame:
        """Convert params_fit to proper DF with columns. Add physical x, y pos.

        Returns:
            pd.DataFrame: [description]
        """
        d = pd.DataFrame(data=self.gabor.params_fit, columns=GaborFit.KEY.keys())
        d.rename(columns=dict(pos_x="azimuth", pos_y="altitude"), inplace=True)
        return d.join(self.loader.pos)

    def prepare_df(self, V2_size: float = 0.4, V2_cutoff: int = 180) -> pd.DataFrame:
        """Add analysis-specific details to df_all. Here, we
            (1) We split V1 and V2 neurons using a straight line in the y-axis. Col: region
            (2) Sample V1 and V2 neurons such that the Gabor x (proxy for overall similarity) matches that of V1s.
            (3) Randomly split V1 and V2 each into 2 groups. For inter and intra-region CCA.

        Args:
            V2_size (float, optional): The number of sampled V2 neurons. Defaults to 0.4.
            V2_cutoff (int, optional): The y-cutoff value for V2. Defaults to 180.

        Returns:
            pd.DataFrame: DF with additional columns {sampled, region, group}
        """
        df_all = self.df_all.copy()

        # Line separating V1 and V2.
        df_all["region"] = pd.Categorical(np.where(df_all.y > V2_cutoff, "V1", "V2"))
        n_V2 = int(V2_size * df_all.groupby("region").size()["V2"])

        # Sample
        df_all["sampled"] = False
        df_all.loc[
            self._match_dist_region(df_all, "azimuth", "V1", "V2", size=n_V2), "sampled"
        ] = True  # V2
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

    def _match_dist_region(
        self, df: pd.DataFrame, param: str, source: str, target: str, size: int
    ) -> np.ndarray:
        """Sample neurons from region `target` such that the distribution of `param` matches that of `source`.

        Args:
            df (pd.DataFrame): DF if Gabor params with region info.
            param (str): Name of Gabor param to use.
            source (str): Source region.
            target (str): Target region.
            size (int): Number of neurons to sample from target.
                        Trade-off between number of neurons and KL divergence.

        Returns:
            np.ndarray: Indices of sampled target neurons.
        """
        assert source in df.region.cat.categories
        assert target in df.region.cat.categories

        val = {reg: df[df.region == reg][param] for reg in [source, target]}
        kde = {reg: gaussian_kde(z) for reg, z in val.items()}
        p = (
            p_unnorm := kde[source].evaluate(val[target]) / kde[target].evaluate(val[target])
        ) / np.sum(p_unnorm)

        rand = np.random.default_rng(self.seed)
        return rand.choice(df[df.region == target].index, size=int(size), replace=False, p=p)

    @staticmethod
    def _gen_idxs_neuron(
        df: pd.DataFrame, region_pair: Tuple[Dict[str, Any], ...]
    ) -> List[pd.Index]:
        """
        Return the indices of neurons as filtered by the the dicts in `region_pair`.

        Args:
            df (pd.DataFrame)
            pair (Tuple[Dict[str, Any], ...]): A tuple of dicts that specifies filters {column: value}.
                Multiple filters are treated as intersections (AND). See `self.regions` for example.

        Returns:
            List[pd.Index]: List of neuron indices.
        """
        return [df.loc[df[p.keys()].isin(p.values()).all(axis=1), :].index for p in region_pair]

    @overload
    def run_cca(
        self, idx_train: np.ndarray, idx_test: Optional[np.ndarray], return_obj: Literal[False]
    ) -> pd.DataFrame:
        ...

    @overload
    def run_cca(
        self, idx_train: np.ndarray, idx_test: Optional[np.ndarray], return_obj: Literal[True]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ...

    def run_cca(self, idx_train, idx_test=None, return_obj=False):
        """
        Run CCA with regions as specified in `self.regions`.
        Canonical vectors are generated from `self.S[idx_stim_train]`.
        Canonical coefs are from both `self.S[idx_stim_train]` and `self.S[idx_stim_test]`.

        Args:
            idx_train (ndarray): Train indices.
            idx_test (Optional[np.ndarray]): Test indices.

        Returns:
            if not return_obj:
                pd.DataFrame: canonical coefs in a tidy df. Columns are [dimension, coefs, regions, split].
            else:
                Above and pd.DataFrame of CanonicalRidge objects with cols [cr, regions].
        """

        out: List[pd.DataFrame] = list()
        cr_obj: List[CanonicalRidge] = list()
        spks_source = (
            self.S if self.spks_source is None else self.spks_source
        )  # For set_spks_source.

        for name, region_pair in self.regions.items():
            # Get neuron indices.
            idxs_neu = self._gen_idxs_neuron(self.df, region_pair)
            region1, region2 = [spks_source[:, idx] for idx in idxs_neu]
            cr = CanonicalRidge(self.n_cc, lambda_x=0.85, lambda_y=0.85).fit(
                region1[idx_train], region2[idx_train]
            )
            out.append(pd.DataFrame(cr.coef).assign(regions=name, split="train"))
            if idx_test is not None:
                out.append(
                    pd.DataFrame(
                        cr.calc_canon_coef(region1[idx_test], region2[idx_test])
                    ).assign(regions=name, split="test")
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
        self, cr: CanonicalRidge, regions: str, idx_test: np.ndarray, stim_idx: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the canonical variates of given `idx_test` between `regions` pair.

        Args:
            cr (CanonicalRidge)
            regions (str): Name of region pairs from `self.regions`.
            idx_test (np.ndarray): Stim indices to calculate.
            stim_idx (bool, optional): Use filtered stim (S) index.
                Otherwise use index that includes spont timepoints. Defaults to True.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Pair of canonical variates.
        """

        region_pair = self.regions[regions]
        idxs_neu = self._gen_idxs_neuron(self.df, region_pair)

        if self.spks_source is None:  # For set_spks_source.
            spks_source = self.S if stim_idx else self.spks
        else:
            spks_source = self.spks_source

        return cr.transform(*[spks_source[idx_test][:, i] for i in idxs_neu])

    @staticmethod
    def pairwise_inner_prod(arr1: np.ndarray, arr2: np.ndarray, normalize=True) -> np.ndarray:
        """Pearson's correlation coefficient for each column of a matrix.

        Args:
            arr1 (np.ndarray)
            arr2 (np.ndarray)

        Returns:
            np.ndarray: Vector containing correlation coefficient.
        """
        assert arr1.shape == arr2.shape
        use = [arr - np.mean(arr, axis=0, keepdims=True) for arr in [arr1, arr2]]
        if normalize:
            use = [arr / norm(arr, axis=0, keepdims=True) for arr in use]
        return np.sum(use[0] * use[1], axis=0)

    def run_random_splits(self, ns: List[int], test_size=0.2) -> pd.DataFrame:
        """Run CCA using `ns` stim points. Test from `train_test_split`.

        Args:
            n_train (List[int])
            test_size (float, optional): Defaults to 0.2.

        Returns:
            pd.DataFrame: DF with col [dimension, coefs, regions, split, n] (from self.run_cca.)
        """
        res = dict()
        for n in ns:
            idx_stim_tr, idx_stim_te = train_test_split(
                np.arange(0, int(n)), test_size=test_size, random_state=self.seed
            )
            res[n] = self.run_cca(idx_train=idx_stim_tr, idx_test=idx_stim_te).assign(n=n)

        return pd.concat(res.values())

    @contextmanager
    def set_spks_source(self, spks_source: np.ndarray) -> None:
        """Override `self.S` as the data matrix.

        Args:
            spks_source (np.ndarray): Data matrix of (n_stim, n_neu).
        """
        self.spks_source = spks_source
        try:
            yield
        finally:
            self.spks_source = None


class CCARepeatedStim(CCARegions):
    """ Class to focus more on the impacts of different stimuli.
    All indices mentioned here are stim indices.
    All neuron indices should be dealt with in `CCARegions`.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Generate `idx_test` and `idx_unrepeated`.
        `idx_test` is the time when the second repeat of the repeated stimuli is shown.
        `idx_unrepeated` is the time when stimuli that are shown once is shown.
        """
        super().__init__(*args, **kwargs)
        self.idx_test = self.loader.get_idx_rep()[:, 1]
        self.idx_unrepeated = np.where(
            np.isin(range(self.S.shape[0]), self.loader.get_idx_rep().flatten(), invert=True)
        )[0]

    def _get_idx_train_repeated(self, n: int) -> pd.DataFrame:
        """Generate indices of training data that include and does not include test stimuli.
        
        if test_stim_in_train:
            Get the first repeat until n>entire repeat set. Then concat some random unrepeated idx.
        else:
            Get random unrepeated idx.
        
        Args:
            n (int): Size of training data.
        Returns:
            pd.DataFrame: DF of indices with cols [stim, test_stim_in_train].
        """
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

    def run_repeated_splits(self, ns_train: List[int]) -> pd.DataFrame:
        """Run CCA to compare the effects of the presence of test stimuli in the training data.

        Args:
            n_train (List[int]): List of number of training samples to run.

        Returns:
            pd.DataFrame: Canonical coefs with col [dimension, coefs, regions, split, n] (from self.run_cca).
        """
        res = list()
        for n in ns_train:
            idx = self._get_idx_train_repeated(n)
            for case in [True, False]:
                coefs = self.run_cca(
                    idx_train=idx[idx.test_stim_in_train.eq(case)]["stim"],
                    idx_test=self.idx_test,
                ).assign(test_stim_in_train=case, n=n)
                res.append(coefs)

        return pd.concat(res)

    def get_cr_unrepeated(self, ns_train: List[int]) -> pd.DataFrame:
        """Return DataFrame consisting of CanonicalRidge objects trained from unrepeated stimuli.

        Args:
            n_train (List[int]): List of number of training samples to run.

        Returns:
            pd.DataFrame: DF with columns [cr, regions, n].
        """
        rand = np.random.default_rng(self.seed)
        res = list()
        for n in ns_train:
            _, objs = self.run_cca(
                idx_train=rand.choice(self.idx_unrepeated, size=n, replace=False),
                return_obj=True,
            )
            res.append(objs.assign(n=n))
        return pd.concat(res)

    def calc_innerprod_test(
        self,
        df_cr: pd.DataFrame,
        idxs_test: Dict[str, np.ndarray],
        pairs: List[Tuple[str, str]],
        normalize: bool = True,
        **run_cca_transform_kwargs,
    ) -> pd.DataFrame:
        """Get df from run_cca and calc correlation between each pair of test indices.

        Args:
            df_cr (pd.DataFrame): df with **at least** cols {cr, regions}
            idxs_test (Dict[str, np.ndarray]): Stim. {name of idx group: idx}
            pairs (List[Tuple[str, str]]): List of comparisons, order-sensitive in each tuple.
            normalize (bool): True means correlation coefficient. False means covariance.

        Returns:
            pd.DataFrame: df with all cols from `df_cr` except `cr` + {match, corr} in tidy format.
        """
        res = list()
        name = "corr" if normalize else "cov"
        for (x, y) in pairs:  # Check validity.
            assert x in idxs_test
            assert y in idxs_test

        for row in df_cr.iloc:
            # Compute variates
            variates = {
                name: self.run_cca_transform(
                    row.cr, regions=row.regions, idx_test=idx, **run_cca_transform_kwargs
                )
                for name, idx in idxs_test.items()
            }
            # Calculate corr.
            corrs = {
                f"{p1}_{p2}": self.pairwise_inner_prod(
                    variates[p1][0], variates[p2][1], normalize=normalize
                )
                for (p1, p2) in pairs
            }
            res.append(
                pd.DataFrame(corrs)
                .reset_index()
                .melt(id_vars="index", var_name="match", value_name=name)
                .assign(**dict(row))
            )

        return pd.concat(res).rename(columns={"index": "dimension"}).drop(columns=["cr"])

    def calc_repeated_corr(self, n: int) -> pd.DataFrame:
        """Calculate the correlation between the first and second repeats of the repeated stimuli.
        For testing.

        Args:
            n (int): number of samples to get from `self._get_idx_train_repeated`.

        Returns:
            pd.DataFrame: DF from process_df with col [regions, group, test_stim_in_train, n]
        """
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
    df_un = cr.get_cr_unrepeated(ns_train=n_train[:3])
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

