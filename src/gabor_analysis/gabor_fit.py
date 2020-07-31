import time
from functools import partial
from importlib import import_module
from typing import Dict, Tuple, Callable

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import logging
import jax
import pandas as pd
from jax import value_and_grad, grad, jit
from jax.numpy import cos, sin, exp
from jax.numpy import pi as π
from jax.random import PRNGKey, randint

from .utils_jax import correlate, zscore_img
from ..analyzer import Analyzer
from ..receptive_field.rf import gen_rf_rank
from ..utils.io import hdf5_load


class GaborFit(Analyzer):
    HYPERPARAMS = ["n_iter", "n_pc", "optimizer", "params_init", "penalties"]
    ARRAYS = ["rf_pcaed", "rf_fit", "corr"]
    DATAFRAMES = ["params_fit"]
    KEY = {s: i for i, s in enumerate(["σ", "θ", "λ", "γ", "φ", "pos_x", "pos_y"])}

    def __init__(
        self,
        n_pc: int = 30,
        n_iter: int = 1500,
        n_split: int = 4,
        optimizer: Dict[str, str] = None,
        params_init: Dict[str, float] = None,
        penalties: np.ndarray = None,
        **kwargs,
    ):
        # Optimizer. See https://jax.readthedocs.io/en/latest/jax.experimental.optimizers.html.
        super().__init__(**kwargs)
        if optimizer is None:
            raise ValueError("Optimizer not named.")
        self.optimizer = optimizer
        self.n_iter = n_iter
        self.n_pc = n_pc
        self.n_split = n_split
        if params_init is None:
            self.params_init = {
                "σ": 1.5,
                "θ": 0.0,
                "λ": 0.9,
                "γ": 1.5,
                "φ": 0.0,
                "pos_x": 0.0,
                "pos_y": 0.0,
            }
        else:
            self.params_init = params_init

        if penalties is None:
            self.penalties = np.zeros((5, 2), dtype=np.float32)
            self.penalties[self.KEY["σ"]] = (0.2, 1.0)
            self.penalties[self.KEY["γ"]] = (0.1, 0.5)
        else:
            self.penalties = penalties
        self.penalties = jnp.array(self.penalties)

    def fit(self, rf: np.ndarray):
        print("Fitting Gabor.")
        assert rf.shape[1] % 2 == 0 and rf.shape[2] % 2 == 0

        if self.n_pc == 0:
            logging.info("No PCA.")
            self.rf_pcaed = rf
        else:
            self.rf_pcaed = gen_rf_rank(rf, self.n_pc)

        opt_funcs = self._get_optimizer(self.optimizer)
        self.rf_pcaed = np.array(zscore_img(self.rf_pcaed))
        rf_dim = jnp.array((self.rf_pcaed.shape[2] // 2, self.rf_pcaed.shape[1] // 2))

        # Split RF.
        splits = [
            i * (self.rf_pcaed.shape[0] - 1) // self.n_split for i in range(self.n_split + 1)
        ]
        splits[-1] = self.rf_pcaed.shape[0] - 1
        n_neu = self.rf_pcaed.shape[0]

        # Prepare outputs.
        self.params_fit = np.zeros((n_neu, len(self.KEY)), dtype=np.float32)
        self.rf_fit = np.zeros(
            (n_neu, self.rf_pcaed.shape[1], self.rf_pcaed.shape[2]), dtype=np.float32
        )
        self.corr = np.zeros(n_neu, dtype=np.float32)

        for i in range(self.n_split):
            sl = np.s_[splits[i] : splits[i + 1]]
            self.params_fit[sl], self.rf_fit[sl], self.corr[sl] = [
                np.array(x) for x in self._split_fit(self.rf_pcaed[sl], opt_funcs, rf_dim, i)
            ]

        self.params_fit = pd.DataFrame(self.params_fit, columns=self.KEY)
        self.params_fit["corr"] = self.corr
        return self

    def _split_fit(
        self,
        rf_pcaed: np.ndarray,
        opt_funcs: Tuple[Callable, ...],
        rf_dim: Tuple[int, int],
        sp: int,
    ) -> Tuple[jnp.DeviceArray, ...]:

        init, update, get_params = opt_funcs
        rf_pcaed = jax.device_put(rf_pcaed, jax.devices()[0])
        params_jax = init(GaborFit._gen_params(rf_pcaed, self.params_init))

        t0 = time.time()
        for i in range(self.n_iter):
            Δ = grad(GaborFit._loss_func)(
                get_params(params_jax), rf_pcaed, rf_dim, self.penalties
            )
            params_jax = update(i, Δ, params_jax)

            if i % 500 == 0 or i == self.n_iter - 1:
                corr = jnp.mean(
                    correlate(self._make_gabor(get_params(params_jax), rf_dim), rf_pcaed)
                )
                logging.info(
                    f"Split {sp}, step {i: 5d}. Corr: {corr: 0.4f} t: {time.time() - t0: 6.2f}s"
                )

        params_fit = get_params(params_jax)
        rf_fit = self._make_gabor(params_fit, rf_dim)
        corr = correlate(rf_fit, rf_pcaed)
        del rf_pcaed

        return params_fit, rf_fit, corr

    @staticmethod
    def _get_optimizer(optimizer):
        opt_func = getattr(import_module("jax.experimental.optimizers"), optimizer["name"])
        optimizer = {k: v for k, v in optimizer.items() if k != "name"}
        optimizer = opt_func(**optimizer)
        init, update, get_params = [jit(f) for f in optimizer]
        return init, update, get_params

    @staticmethod
    @partial(jit, static_argnums=1)
    def _make_gabor(params: jnp.ndarray, rf_dim: Tuple[int, int]) -> jnp.DeviceArray:
        σ, θ, λ, γ, φ = [
            u[:, jnp.newaxis, jnp.newaxis]
            for u in (params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4])
        ]
        pos_x, pos_y = [u[:, jnp.newaxis, jnp.newaxis] for u in (params[:, 5], params[:, 6])]

        n = params.shape[0]

        x, y = jnp.meshgrid(
            jnp.arange(-rf_dim[0], rf_dim[0]), jnp.arange(-rf_dim[1], rf_dim[1])
        )
        x = jnp.repeat(x[jnp.newaxis, :, :], n, axis=0)
        y = jnp.repeat(y[jnp.newaxis, :, :], n, axis=0)

        xp = (pos_x - x) * cos(θ) - (pos_y - y) * sin(θ)
        yp = (pos_x - x) * sin(θ) + (pos_y - y) * cos(θ)

        output = exp(-(xp ** 2 + (γ * yp) ** 2) / (2 * σ ** 2)) * exp(1j * (2 * π * xp / λ + φ))

        return zscore_img(output.real)

    @staticmethod
    @partial(jit, static_argnums=(2, 3))
    def _loss_func(params, img, rf_dim, penalties):
        made = GaborFit._make_gabor(params, rf_dim)
        metric = jnp.mean(-correlate(made, img))

        for i in range(5):
            metric += penalties[i, 0] * jnp.mean(
                jnp.maximum(i, -params[:, i] + penalties[i, 1])
            )

        return metric

    @staticmethod
    def _gen_params(rf, p):
        n = rf.shape[0]

        params = np.zeros((n, len(p)))
        for i, v in enumerate(p.values()):
            params[:, i] = v

        # Center location.
        yc, xc = rf.shape[1] // 2 + 1, rf.shape[2] // 2 + 1

        idx = np.argmax(np.abs(rf.reshape([n, -1])), axis=1,)
        params[:, 5] = idx % rf.shape[2] - xc  # x
        params[:, 6] = idx // rf.shape[2] - yc  # y

        return params

    def plot(self, seed: int = 40, save: str = None, title=None) -> None:
        rng = PRNGKey(seed)
        idx = sorted(randint(rng, shape=(10,), minval=0, maxval=self.rf_fit.shape[0]))

        fig, axs = plt.subplots(
            nrows=4, ncols=5, figsize=(10, 6), dpi=300, constrained_layout=True
        )
        axs = np.hstack([axs[0:2, :], axs[2:4, :]]).T

        for i in range(10):
            scale = jnp.max(jnp.abs(self.rf_fit[idx[i], ...]))
            axs[i, 0].imshow(
                self.rf_fit[idx[i], ...], cmap="twilight_shifted", vmin=-scale, vmax=scale
            )
            axs[i, 0].set_title(f"Neuron {idx[i]}\n" f"Pearson's r: {self.corr[idx[i]]: 0.3f}")
            axs[i, 0].axis("off")

            scale = jnp.max(jnp.abs(self.rf_pcaed[idx[i], ...]))
            axs[i, 1].imshow(
                self.rf_pcaed[idx[i], ...], cmap="twilight_shifted", vmin=-scale, vmax=scale
            )
            axs[i, 1].axis("off")

        if title is None:
            title = f"Randomly selected 10 neurons from {self.rf_fit.shape[0]} neurons. "
            f"\n Top {self.n_pc} PCs RF with corr coef."

        fig.suptitle(title)

        if save:
            plt.savefig(save)
        plt.show()


def make_regression_truth():
    rf = hdf5_load("tests/data/regression_test_data.hdf5", "ReceptiveField", arrs=["neu"])[
        "neu"
    ]
    gabor = GaborFit(n_pc=30, n_iter=500, optimizer={"name": "adam", "step_size": 2e-2}).fit(rf)
    gabor.plot()
    gabor.save_append("tests/data/regression_test_data.hdf5")

