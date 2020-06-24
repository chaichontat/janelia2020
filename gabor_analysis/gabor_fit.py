import pickle
import time
from functools import partial
from pathlib import Path
from typing import Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import value_and_grad, grad, jit
from jax.experimental.optimizers import adam
from jax.numpy import cos, sin, exp
from jax.numpy import pi as π
from jax.random import PRNGKey, randint

from gabor_analysis.receptive_field import reduce_rf_rank
from gabor_analysis.utils_jax import correlate, zscore_img


class GaborFit:

    def __init__(self, n_pc=30, n_iters=1500, img_dim=(16, 9), optimizer=adam(1e-2)):
        self.img_dim = img_dim
        self.optimizer = optimizer
        self.n_iters = n_iters
        self.n_pc = n_pc

        # Filled with self.pca and self.fit.
        self.rf_raw = jnp.empty(0)
        self.rf_pcaed, self.rf_pcs = jnp.empty(0), jnp.empty(0)

        # Filled with self.fit.
        self.rf_fit, self.params, self.corr = jnp.empty(0), jnp.empty(0), jnp.empty(0)

    def pca_rf(self, B):
        self.rf_raw = B
        self.rf_pcaed, self.rf_pcs = reduce_rf_rank(B, self.n_pc)
        return self

    def fit(self, B):
        print('Fitting Gabor.')
        self.rf_raw = B
        if self.n_pc == 0:
            print('No PCA.')
            self.rf_pcaed = B
        else:
            self.rf_pcaed, self.rf_pcs = reduce_rf_rank(B, self.n_pc)

        self.rf_pcaed = zscore_img(self.rf_pcaed)

        init, update, get_params = [jit(f) for f in self.optimizer]
        params_jax = init(GaborFit._gen_params(self.rf_pcaed))

        t0 = time.time()
        for i in range(self.n_iters // 3):
            if i % 100 == 0:
                loss, Δ = value_and_grad(GaborFit._loss_func)(get_params(params_jax), self.rf_pcaed)
                corr = jnp.mean(correlate(self._make_gabor(self.img_dim, get_params(params_jax)), self.rf_pcaed))
                print(f'Step {3 * i} Corr: {corr: 0.4f} t: {time.time() - t0: 6.2f}s')
                params_jax = update(i, Δ, params_jax)
            else:
                params_jax = GaborFit._jax_fit(params_jax, self.rf_pcaed, get_params, update)

        self.params = get_params(params_jax)
        self.rf_fit = self._make_gabor(self.img_dim, self.params)
        self.corr = correlate(self.rf_fit, self.rf_pcaed)

        return self

    @staticmethod
    @partial(jit, static_argnums=(2, 3))
    def _jax_fit(p, img, get_params, update):
        for i in range(3):
            Δ = grad(GaborFit._loss_func)(get_params(p), img)
            p = update(i, Δ, p)
        return p

    @staticmethod
    @partial(jit, static_argnums=0)
    def _make_gabor(size: Tuple[int, int], params: jnp.ndarray) -> jnp.ndarray:
        σ, θ, λ, γ, φ = [u[:, jnp.newaxis, jnp.newaxis] for u in
                         (params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4])]
        pos_x, pos_y = [u[:, jnp.newaxis, jnp.newaxis] for u in (params[:, 5], params[:, 6])]

        n = params.shape[0]

        x, y = jnp.meshgrid(jnp.arange(-size[0], size[0]), jnp.arange(-size[1], size[1]))
        x = jnp.repeat(x[jnp.newaxis, :, :], n, axis=0)
        y = jnp.repeat(y[jnp.newaxis, :, :], n, axis=0)

        xp = (pos_x - x) * cos(θ) - (pos_y - y) * sin(θ)
        yp = (pos_x - x) * sin(θ) + (pos_y - y) * cos(θ)

        output = exp(-(xp ** 2 + (γ * yp) ** 2) / (2 * σ ** 2)) * exp(1j * (2 * π * xp / λ + φ))

        return zscore_img(output.real)

    @staticmethod
    @jit
    def _loss_func(params, img):
        size = (16, 9)
        made = GaborFit._make_gabor(size, params)
        metric = jnp.mean(-correlate(made, img))

        penalty_σ = 0.2 * jnp.mean(jnp.maximum(0, -params[:, 0] + 1))  # flipped ReLU from 1
        # penalty_θ = np.mean((2 * params[:, 2])**4)  # np.mean(np.maximum(0, params[:, 1] - 0.4))  # flipped ReLU from 1
        # penalty_λ = 0.1 * np.mean((params[:, 2] - 1.)**2)
        penalty_γ = 0.1 * jnp.mean(jnp.maximum(0, -params[:, 3] + 0.5))

        return metric + penalty_σ + penalty_γ  # + penalty_θ #+ penalty_γ

    @staticmethod
    def _gen_params(rf):
        p = {
            'σ': 1.5,
            'θ': 0.,
            'λ': 1.,
            'γ': 1.5,
            'φ': 0.,
            'pos_x': 0.,
            'pos_y': 0.,
        }
        n = rf.shape[0]

        params = np.zeros((n, len(p)))
        for i, v in enumerate(p.values()):
            params[:, i] = v

        # Center location.
        yc, xc = rf.shape[1] // 2 + 1, rf.shape[2] // 2 + 1

        idx = np.argmax(np.abs(rf.reshape([n, -1])), axis=1, )
        params[:, 5] = idx % rf.shape[2] - xc  # x
        params[:, 6] = idx // rf.shape[2] - yc  # y

        return params

    def plot(self, seed: int = 40, save: str = None) -> None:
        rng = PRNGKey(seed)
        idx = randint(rng, shape=(10,), minval=0, maxval=self.rf_fit.shape[0])

        fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(10, 6), dpi=300, constrained_layout=True)
        axs = np.hstack([axs[0:2, :], axs[2:4, :]]).T

        for i in range(10):
            scale = jnp.max(jnp.abs(self.rf_fit[idx[i], ...]))
            axs[i, 0].imshow(self.rf_fit[idx[i], ...], cmap='bwr', vmin=-scale, vmax=scale)
            axs[i, 0].set_title(f'Neuron {idx[i]}\n'
                                f'Pearson\'s r: {self.corr[idx[i]]: 0.3f}')
            axs[i, 0].axis('off')

            scale = jnp.max(jnp.abs(self.rf_pcaed[idx[i], ...]))
            axs[i, 1].imshow(self.rf_pcaed[idx[i], ...], cmap='bwr', vmin=-scale, vmax=scale)
            axs[i, 1].axis('off')

        fig.suptitle(
            f'Randomly selected 10 neurons from {self.rf_fit.shape[0]} neurons. '
            f'\n Top {self.n_pc} PCs RF with corr coef. L: Generated, R: Raw.')

        if save:
            plt.savefig(save)
        plt.show()


if __name__ == '__main__':
    B = pickle.loads(Path(f'gabor_analysis/field.pk').read_bytes())
    gabor = GaborFit().fit(B)
    gabor.plot()
