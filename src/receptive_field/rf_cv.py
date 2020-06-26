from sklearn.metrics import mean_squared_error
from skopt import gp_minimize
from skopt.space import Real

from .rf import ReceptiveField
from ..spikeloader import SpikeLoader

# %% CV λ
loader = SpikeLoader()
trX, teX, trS, teS = loader.train_test_split()


def objective(λ):
    rf = ReceptiveField(loader, λ=λ).fit_neuron(trX, trS)
    S_hat = rf.transform(teX)
    mse = mean_squared_error(teS, S_hat)

    rf = ReceptiveField(loader, λ=λ).fit_neuron(teX, teS)
    S_hat = rf.transform(trX)
    mse += mean_squared_error(trS, S_hat)
    return float(mse)


space = [
    Real(0.1, 100, prior='log-uniform', name='λ')
]

res_gp = gp_minimize(objective, space, n_calls=20, n_random_starts=10, random_state=439,
                     verbose=True)
