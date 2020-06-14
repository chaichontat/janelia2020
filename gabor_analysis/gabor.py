import pickle
import time
from functools import partial
from pathlib import Path

import jax.numpy as np
import matplotlib.pyplot as plt
import numpy as onp
import seaborn as sns
from jax import value_and_grad, grad, jit
from jax.experimental.optimizers import rmsprop_momentum
from jax.numpy import cos, sin, exp
from jax.numpy import pi as π
from jax.random import PRNGKey, randint, normal

from scipy.stats import zscore
from utils import reduce_B_rank
from utils_jax import *


def make_gabor(size, params):
    σ, θ, λ, γ, φ = [u[:, np.newaxis, np.newaxis] for u in (params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4])]
    pos_x, pos_y = [u[:, np.newaxis, np.newaxis] for u in (params[:, 5], params[:, 6])]

    n = params.shape[0]

    x, y = np.meshgrid(np.arange(-size[0], size[0]), np.arange(-size[1], size[1]))
    x = np.repeat(x[np.newaxis, :, :], n, axis=0)
    y = np.repeat(y[np.newaxis, :, :], n, axis=0)

    xp = (pos_x - x) * cos(θ) - (pos_y - y) * sin(θ)
    yp = (pos_x - x) * sin(θ) + (pos_y - y) * cos(θ)

    output = exp(-(xp**2 + (γ*yp)**2) / (2*σ**2)) * exp(1j * (2*π*xp/λ + φ))

    return zscore_img(output.real)

def plot_gabor(data, p, n_pca=None, error_metric=mse, seed=40):
    B_rz = zscore_img(data)
    rng = PRNGKey(seed)
    idx_sorted = randint(rng, shape=(10,), minval=0, maxval=len(B_rz))

    gen = make_gabor((16, 9), p)

    sns.set()
    fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(10, 6), dpi=300, constrained_layout=True)
    axs = onp.hstack([axs[0:2, :], axs[2:4, :]]).T

    error = error_metric(B_rz, gen)

    for i in range(10):
        scale = np.max(np.abs(gen[idx_sorted[i], ...]))
        axs[i, 0].imshow(gen[idx_sorted[i], ...], cmap='bwr', vmin=-scale, vmax=scale)
        axs[i, 0].set_title(f'{error[idx_sorted[i]]: .3f}')

        scale = np.max(np.abs(B_rz[idx_sorted[i], ...]))
        axs[i, 1].imshow(B_rz[idx_sorted[i], ...], cmap='bwr', vmin=-scale, vmax=scale)
        axs[i, 0].axis('off')
        axs[i, 1].axis('off')

    fig.suptitle(
        f'Randomly selected 10 neurons from {len(data)} neurons. \n Top {n_pca} PCs RF with corr coef. L: Generated, R: Raw.')
    plt.show()

@partial(jit)
def error(params, img, size=(16, 9)):
    made = make_gabor(size, params)
    metric = np.mean(-correlate(made, img))

    penalty_σ = np.mean((params[:, 0] < 0.9) * (-params[:, 0] + 0.9))  # flipped ReLU from 1
    penalty_λ = 0.1 * np.mean((params[:, 2] - 1.)**2)
    penalty_γ = 0.2 * np.mean((params[:, 3] < 0.5) * (-params[:, 3] + 0.5))

    return metric + penalty_σ + penalty_λ + penalty_γ

def gen_start(img):
    p = {
        'σ': 1.,
        'θ': 0.,
        'λ': 1.,
        'γ': 1.5,
        'φ': 0.,
        'pos_x': 0.,
        'pos_y': 0.,
    }
    n = img.shape[0]

    params = onp.zeros((n, len(p)))
    for i, v in enumerate(p.values()):
        params[:, i] = v

    # Center location.
    yc, xc = img.shape[1] // 2 + 1, img.shape[2] // 2 + 1

    idx = onp.argmax(onp.abs(img.reshape([n, -1])), axis=1,)
    params[:, 5] = idx % img.shape[2] - xc  # x
    params[:, 6] = idx // img.shape[2] - yc  # y

    return params

@jit
def jax_fit(params, test, update, get_params):
    for i in range(3):
        Δ = grad(error)(get_params(params), test)
        params = update(i, Δ, params)
    return params


def fit(B, rank):
    B_reduced, pcs = reduce_B_rank(B, rank)
    B_rz = zscore_img(B_reduced)

    params_jax = gen_start(B_rz)

    init, update, get_params = [jit(f) for f in rmsprop_momentum(5e-4, momentum=0.99)]
    params_jax = init(params_jax)

    t0 = time.time()
    for i in range(2000):
        if i % 100 == 0:
            corr, Δ = value_and_grad(error)(get_params(params_jax), B_rz)
            print(f'Step {3 * i} Corr: {-corr: 0.4f} t: {time.time() - t0: 6.2f}s')
            params_jax = update(i, Δ, params_jax)
        else:
            params_jax = jax_fit(params_jax, B_rz, update, get_params)

    return B_rz, get_params(params_jax)


#%% Synthetic
def validate():
    params_real = {
        'σ': 5.,
        'θ': π / 4,
        'λ': π / 4,
        'γ': 0.8,
        'φ': 0.1,
        'pos_x': 0.,
        'pos_y': 5.,
        'scale': 1.,
        'b': 0.
    }

    params_real = np.array([[v for k, v in params_real.items()]])

    rng = PRNGKey(5)
    gnd = zscore(make_gabor((30, 20), params_real) + 0.1 * normal(rng, shape=(40, 60)))
    gnd = np.repeat(gnd, 5, axis=0)

    B_rz, params_jax = fit(gnd, 100)

    fig, axs = plt.subplots(ncols=2, figsize=(12, 5), dpi=300)
    axs[0].imshow(gnd[0, ...])
    axs[1].imshow(make_gabor((30, 20), params_jax)[0, ...])
    gen_start(onp.asarray(gnd))
    plt.show()


if __name__ == '__main__':
    run = True
    if run:
        B = pickle.loads(Path(f'field.pk').read_bytes())
        n_pca = 60
        B_rz, params_jax = fit(B, n_pca)

        with open(f'gabor_{n_pca}.pk', 'wb') as f:
            pickle.dump(params_jax, f)

        plot_gabor(B_rz, params_jax)
