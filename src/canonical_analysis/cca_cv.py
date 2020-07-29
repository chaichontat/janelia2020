from functools import partial

import jax.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skopt import gp_minimize
from skopt import plots
from skopt.space import Real

from ..canonical_analysis.canonical_ridge import CanonicalRidge
from ..spikeloader import SpikeLoader

""" 2-fold CV for hyperparameter optimization using Gaussian processes. """


def prepare_train_test(loader: SpikeLoader):
    train, test = loader.train_test_split()

    # Split V1 and V2
    train = [train[:, loader.ypos >= 210], train[:, loader.ypos < 210]]
    test = [test[:, loader.ypos >= 210], test[:, loader.ypos < 210]]
    return train, test


def objective(train, test, *λ):
    print(f'{λ=}')
    λ = λ[0]
    model = CanonicalRidge(lambda_x=λ[0], lambda_y=λ[1]).fit(*train)
    tr = -np.mean(model.calc_canon_coef(*test))

    model = CanonicalRidge(lambda_x=λ[0], lambda_y=λ[1]).fit(*test)
    tr += -np.mean(model.calc_canon_coef(*train))
    return float(tr)


if __name__ == '__main__':
    sns.set()
    loader = SpikeLoader()
    train, test = prepare_train_test(loader)
    X, Y = np.asarray(train[0]), np.asarray(train[1])

    space = [
        Real(0.8, 0.9999, name='λx'),
        Real(0.8, 0.9999, name='λy')
    ]

    func = partial(objective, train, test)
    res_gp = gp_minimize(func, space, n_calls=20, n_random_starts=5, random_state=439,
                         verbose=True)

    plt.rcParams['figure.dpi'] = 300
    plots.plot_objective(res_gp, size=4)
    plt.suptitle('Gaussian Process λ tuning. \n'
                 'Metric is -sum of mean canonical correlation coeffs on test set across 2-folds.')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

# #%%
# model1 = PCA(n_components=30).fit(V1_nocc)
# model2 = PCA(n_components=30).fit(V2_nocc)
#
# #%%
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(10, 6), dpi=300, constrained_layout=True)
#
# axs = np.hstack([axs[0:2, :], axs[2:4, :]]).T
# for i in range(10):
#     scale = np.max(np.abs([B[idx[i], :, :], B_reduced[idx[i], :, :]]))
#     axs[i, 0].imshow(B[idx[i], :, :], cmap='twilight_shifted', vmin=-scale, vmax=scale)
#     axs[i, 1].imshow(B_reduced[idx[i], :, :], cmap='twilight_shifted', vmin=-scale, vmax=scale)
#     axs[i, 0].axis('off')
#     axs[i, 1].axis('off')
#
#
# # fig, axs = plt.subplots(nrows=2, ncols=2, dpi=300, figsize=(10, 8), constrained_layout=True)
# #
# # axs[0, 0].scatter(tr1[:, 0], tr2[:, 0], s=2, alpha=0.5, linewidth=0)
# # axs[0, 0].set_title(f'Train: first CC, Corr: {onp.corrcoef(tr1[:, 0], tr2[:, 0])[0, 1]:.3f}')
# # axs[0, 1].scatter(tr1[:, 1], tr2[:, 1], s=2, alpha=0.5, linewidth=0)
# # axs[0, 1].set_title(f'Train: second CC, Corr: {onp.corrcoef(tr1[:, 1], tr2[:, 1])[0, 1]:.3f}')
# # axs[1, 0].scatter(te1[:, 0], te2[:, 0], s=2, alpha=0.5, linewidth=0)
# # axs[1, 0].set_title(f'Test: second CC, Corr: {onp.corrcoef(te1[:, 0], te2[:, 0])[0, 1]:.3f}')
# # axs[1, 1].scatter(te1[:, 1], te2[:, 1], s=2, alpha=0.5, linewidth=0)
# # axs[1, 1].set_title(f'Test: second CC, Corr: {onp.corrcoef(te1[:, 1], te2[:, 1])[0, 1]:.3f}')
# #
# # axs = axs.flatten()
# # for ax in axs:
# #     ax.set_xlabel('V1')
# #     ax.set_ylabel('V2')
