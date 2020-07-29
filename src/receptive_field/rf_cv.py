from sklearn.metrics import mean_squared_error
from skopt import gp_minimize
from skopt.space import Real, Integer
from scipy.stats import zscore
from sklearn.model_selection import KFold

from .rf import ReceptiveField, ReducedRankReceptiveField
from ..spikeloader import SpikeLoader
import numpy as np

# %% CV λ

def rf_cv(loader, n_splits=5):

    kf = KFold(n_splits=n_splits)

    def objective(λ):
        score = 0
        for tr_idx, te_idx in kf.split(loader.imgs_stim):
            trX, teX = loader.imgs_stim[tr_idx], loader.imgs_stim[te_idx]
            trS, teS = loader.S[tr_idx], loader.S[te_idx]

            rf = ReceptiveField(loader.img_dim, lamda=λ).fit_neuron(trX, trS)
            teS_hat = rf.transform(teX)
            score += mean_squared_error(teS_hat, teS)
        return float(score / n_splits)


    space = [
        Real(-1, 3, prior='uniform', name='λ')
    ]

    res_gp = gp_minimize(objective, space, n_calls=20, n_random_starts=10, random_state=439,
                         verbose=True)

    return res_gp.x


def rr_rf_cv(loader, n_splits=3):

    kf = KFold(n_splits=n_splits)

    def objective(λ):
        print(λ)
        λ, rank = λ
        score = 0
        for tr_idx, te_idx in kf.split(loader.imgs_stim):
            trX, teX = loader.imgs_stim[tr_idx], loader.imgs_stim[te_idx]
            trS, teS = loader.S[tr_idx], loader.S[te_idx]

            rf = ReducedRankReceptiveField(loader.img_dim, lamda=λ, rank=rank).fit_neuron(trX, trS)
            teS_hat = rf.transform(teX)
            score += mean_squared_error(teS_hat, teS)
        return float(score / n_splits)


    space = [
        Real(0.001, 100, prior='log-uniform', name='λ'),
        Integer(3, 300, prior='uniform', name='rank')
    ]

    return gp_minimize(objective, space, n_calls=20, n_random_starts=15, random_state=439,
                         verbose=True, noise=1e-10)

if __name__ == '__main__':
    loader = SpikeLoader.from_hdf5()
    test = rr_rf_cv(loader)

    from skopt.plots import plot_objective
    import matplotlib.pyplot as plt

    plot_objective(test)
    plt.tight_layout()
    plt.show()