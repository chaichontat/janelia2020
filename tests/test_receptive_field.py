from datetime import timedelta
from functools import partial

import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from hypothesis import settings

from src.receptive_field.rf import ReceptiveField
from src.spikeloader import SpikeLoader
from src.utils.io import hdf5_load

settings.register_profile('default', max_examples=10, deadline=timedelta(milliseconds=20000))
settings.load_profile('default')

posints = partial(st.integers, min_value=1)


@given(img_dim=st.tuples(posints(max_value=100), posints(max_value=100)), n=posints(max_value=10000))
def test_reshape_rf(img_dim, n):
    x = np.random.rand(n, *img_dim)
    coef = x.reshape([n, -1]).T
    assert np.allclose(ReceptiveField.reshape_rf(coef, img_dim, smooth=0), x)


@given(st.tuples(st.integers(2, 6), st.integers(2, 6)), st.integers(100, 200), st.integers(1, 30),
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
    for i in range(n_neu):  # Prevent all 0s.
        β[i, np.random.randint(0, img_dim[0] * img_dim[1])] = 1

    spks = X @ β.T
    spks += 0.01 * np.mean(spks) * np.random.normal(size=spks.shape)

    x = ReceptiveField(img_dim)
    x.fit_neuron(X, spks)
    print(np.corrcoef(x.rf_.flatten(), β.flatten())[0, 1])
    assert np.corrcoef(x.rf_.flatten(), β.flatten())[0, 1] > 0.95


def test_fit_regression():
    loader = SpikeLoader.from_hdf5('tests/data/processed.hdf5')
    gnd = hdf5_load('tests/data/regression_test_data.hdf5', 'ReceptiveField', arrs=['neu', 'pc'])

    rf = ReceptiveField(loader.img_dim)
    rf.fit_neuron(loader.imgs_stim, loader.S)
    assert np.allclose(gnd['neu'], rf.rf_, atol=1e-4, rtol=1e-2)

    rf.fit_pc(loader.imgs_stim, loader.S)
    assert np.allclose(gnd['pc'], rf.rf_, atol=1e-4, rtol=1e-2)
