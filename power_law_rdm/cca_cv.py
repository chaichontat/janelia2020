from functools import partial

import jax.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skopt import gp_minimize
from skopt import plots
from skopt.space import Real

from cca import CanonicalRidge, prepare_data

""" 2-fold CV for hyperparameter optimization using Gaussian processes. """

sns.set()
train, test = prepare_data()


# %% Run optimization
def objective(train, test, *λ):
    print(f'{λ=}')
    λ = λ[0]
    model = CanonicalRidge(lx=λ[0], ly=λ[1]).fit(*train)
    tr = -np.mean(model.calc_canon_coef(*test))

    model = CanonicalRidge(lx=λ[0], ly=λ[1]).fit(*test)
    tr += -np.mean(model.calc_canon_coef(*train))
    return float(tr)

space = [
    Real(0.8, 0.9999, name='λx'),
    Real(0.8, 0.9999, name='λy')
]

func = partial(objective, train, test)
res_gp = gp_minimize(func, space, n_calls=20, n_random_starts=5, random_state=439,
                     verbose=True)

# %%
plt.rcParams['figure.dpi'] = 300
plots.plot_objective(res_gp, size=4)
plt.suptitle('Gaussian Process λ tuning. \n'
             'Metric is -sum of mean canonical correlation coeffs on test set across 2-folds.')
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()
