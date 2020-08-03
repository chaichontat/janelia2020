# %%
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.canonical_analysis.canonical_ridge import CanonicalRidge
from src.gabor_analysis.gabor_fit import GaborFit
from src.receptive_field.rf import ReceptiveField, gen_rf_rank
from src.spikeloader import SpikeLoader
from src.analyzer import load_if_exists
from src.gabor_analysis.gabor_fit import correlate

sns.set()
logging.getLogger().setLevel(logging.INFO)


def preprocess(in_file="data/superstim32.npz", out_file=None):
    in_file = Path(in_file)
    loader = SpikeLoader.from_npz(in_file, img_scale=0.25)

    if out_file is None:
        loader.save(in_file.parent / f"{in_file.stem}.hdf5", overwrite=True)
        loader.save_processed(in_file.parent / f"{in_file.stem}_proc.hdf5", overwrite=True)
    else:
        loader.save(out_file, overwrite=True)

    return loader


def visualize_x(loader: SpikeLoader, axis="x", thr=-0.5e-5):
    x_loc = np.expand_dims(loader.pos[axis], axis=1)
    u = loader.pos[axis]
    x = np.linspace(u.min(), u.max(), 10000)

    from sklearn.neighbors import KernelDensity

    kde = KernelDensity(kernel="gaussian", bandwidth=4).fit(x_loc)
    density = np.exp(kde.score_samples(np.expand_dims(x, axis=1)))
    from scipy.signal import find_peaks

    diff = np.diff(density)
    plt.plot(x[:-1], diff)
    dist = find_peaks(-np.clip(diff, -np.inf, thr))[0] / 10000 * np.ptp(u) + np.min(u)
    return np.min(u), dist


@load_if_exists(ReceptiveField)
def run_receptive_field(loader, out_file="data/superstim_TX57_proc.hdf5", overwrite=False):
    logging.info("Running receptive field.")
    rf = ReceptiveField(loader.img_dim, lamda=1.1)
    rf.fit_neuron(loader.imgs_stim, loader.S)
    rf.save_append(out_file, overwrite_node=True)
    return rf


@load_if_exists(GaborFit)
def run_gabor(rf: np.ndarray, penalties, out_file="data/gabor.hdf5", overwrite=False):
    logging.info("Running Gabor.")
    g = GaborFit(
        n_pc=0,
        optimizer={"name": "adam", "step_size": 2e-2},
        params_init={
            "σ": 2,
            "θ": 0.0,
            "λ": 1.0,
            "γ": 1.5,
            "φ": 0.0,
            "pos_x": 0.0,
            "pos_y": 0.0,
        },
        penalties=penalties,
    ).fit(rf)
    g.save_append(out_file, overwrite_node=True)
    return g


def pca_this_data(loader: SpikeLoader, rf: ReceptiveField, plot: bool = False):
    pos = loader.pos
    x_cuts = [pos["x"].min() + i * 650 for i in range(6)]
    y_cuts = [pos["y"].min() + i * 1653 for i in range(3)]
    x_cuts[-1] = pos["x"].max()
    y_cuts[-1] = pos["y"].max()

    rf_pcaed = np.inf * np.ones(rf.rf_.shape, dtype=np.float32)  # type: ignore
    for i in range(len(x_cuts) - 1):
        for j in range(len(y_cuts) - 1):
            idx = pos[pos.x.between(x_cuts[i], x_cuts[i + 1]) & pos.y.between(y_cuts[j], y_cuts[j + 1])].index  # type: ignore
            rf_pcaed[idx] = gen_rf_rank(rf.rf_[idx], 50)
    assert np.all(np.isfinite(rf_pcaed))
    # not_captured = loader.pos.iloc[np.argwhere(~np.isfinite(np.max(rf_pcaed, axis=(1, 2)))).squeeze()]
    if plot:
        pos = loader.pos
        plt.scatter(pos.x, pos.y, s=0.8, alpha=0.5)
        plt.axhline(pos.y.min() + 1653)
        [plt.axvline(pos["x"].min() + i * 650) for i in range(6)]
        plt.show()

    return rf_pcaed


if __name__ == "__main__":
    file = "data/superstim_TX57_proc.hdf5"

    def penalties(name):
        out = np.zeros((5, 2), dtype=np.float32)
        if name == "data/superstim_TX57_proc.hdf5":
            out[GaborFit.KEY["σ"]] = (0.1, 2.5)
            out[GaborFit.KEY["λ"]] = (0.5, 0.85)
            out[GaborFit.KEY["γ"]] = (0.5, 0.5)
        elif name == "data/superstim_TX56_proc.hdf5":
            out[GaborFit.KEY["σ"]] = (0.04, 2.0)
            out[GaborFit.KEY["λ"]] = (0.6, 0.85)
            out[GaborFit.KEY["γ"]] = (0.8, 0.5)
        return out

    loader = SpikeLoader.from_hdf5(file)
    rf = run_receptive_field(loader, out_file=file)
    rf_pcaed = pca_this_data(loader, rf)
    g = run_gabor(rf_pcaed, penalties(file), out_file=file, overwrite=True)

#%% Plot correlation
    pc = correlate(rf.rf_, rf_pcaed)
    ga = correlate(rf.rf_, g.rf_fit)

    plt.scatter(pc, ga, s=1, alpha=0.4)
    plt.show()

#%% Plot bad performers.
    bad = g.params_fit.iloc[np.argwhere((ga < 0.1) & (pc > 0.5)).squeeze()]
    def plot(df, df2=None):
        fig, axs = plt.subplots(figsize=(18, 10), nrows=3, ncols=3)
        axs = axs.flatten()
        for i, column in enumerate(df.columns):
            sns.distplot(df[column], ax=axs[i])
            if df2 is not None:
                sns.distplot(df2[column], ax=axs[i])
            axs[i].set_title(column)
        plt.tight_layout()
        plt.show()

    plot(g.params_fit, bad)
