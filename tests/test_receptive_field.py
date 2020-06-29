from datetime import timedelta
from functools import partial

import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from hypothesis import settings
from sklearn.metrics import mean_squared_error

from src.receptive_field.rf import ReceptiveField

settings.register_profile('default', max_examples=10, deadline=timedelta(milliseconds=20000))
settings.load_profile('default')

posints = partial(st.integers, min_value=1)


@given(img_dim=st.tuples(posints(max_value=100), posints(max_value=100)), n=posints(max_value=10000))
def test_reshape_rf(img_dim, n):
    x = np.random.rand(n, *img_dim)
    rf = ReceptiveField(img_dim)
    rf.coef_ = x.reshape([n, -1]).T
    assert np.allclose(rf._reshape_rf(rf.coef_, smooth=0), x)


@given(st.tuples(st.integers(2, 6), st.integers(2, 6)), st.integers(40, 100), st.integers(1, 30),
       st.integers(0, 1024))
def test_fit_neuron(img_dim, n_t, n_neu, seed):
    np.random.seed(seed)
    rfs = np.repeat(np.zeros(img_dim)[np.newaxis, :], n_neu, axis=0)
    for i in range(n_neu):
        a, b = np.random.randint(0, max(img_dim)), np.random.randint(0, max(img_dim))
        r = np.random.randint(1, round(max(img_dim)))

        y, x = np.ogrid[-a:img_dim[0] - a, -b:img_dim[1] - b]
        mask = x * x + y * y <= r * r
        rfs[i, mask] = 1

    imgs = np.random.rand(n_t, *img_dim)

    X = imgs.reshape([n_t, -1])
    β = rfs.reshape([n_neu, -1])

    spks = X @ β.T
    spks += 0.1 * np.mean(spks) * np.random.normal(size=spks.shape)

    x = ReceptiveField(img_dim)
    x.fit_neuron(X, spks)
    assert mean_squared_error(x.coef_, β) < 0.3
