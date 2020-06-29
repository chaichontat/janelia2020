import hypothesis.strategies as st
import numpy as np
import pandas as pd
from hypothesis import given

from src.spikeloader import SpikeLoader


def test_spikeloader():
    loader = SpikeLoader.from_hdf5('tests/data/raw.hdf5')
    gnd = SpikeLoader.from_hdf5('tests/data/processed.hdf5')
    assert np.allclose(loader.S, gnd.S)
    assert np.allclose(loader.imgs_stim, gnd.imgs_stim)


@given(st.integers(min_value=1, max_value=100), st.integers(min_value=2, max_value=50))
def test_get_idx_rep_rep(length, n_rep):
    loader = SpikeLoader(path=None, pos=None, istim=None, img_scale=None)
    loader.istim = pd.Series(np.repeat(np.arange(length), n_rep, axis=0))
    rep, not_rep = loader.get_idx_rep(return_onetimers=True)
    assert rep.shape == (length, n_rep)
    assert not_rep.size == 0


@given(st.integers(min_value=1, max_value=200))
def test_get_idx_rep_rand(length):
    loader = SpikeLoader(path=None, pos=None, istim=None, img_scale=None)
    loader.istim = pd.Series(np.random.randint(low=0, high=100, size=length))
    rep, not_rep = loader.get_idx_rep(return_onetimers=True)

    count = loader.istim.value_counts()
    assert rep.shape[0] == sum(count > 1)
    assert rep.shape[1] == max(count)
    assert len(not_rep) == sum(count == 1)

    for i in range(rep.shape[0]):
        curr = loader.istim.iloc[rep[i, 0]]
        assert sum(rep[i, :] != -1) == count[curr]
        for j in range(rep.shape[1]):
            if rep[i, j] != -1:
                assert loader.istim.iloc[rep[i, j]] == curr

    assert len(np.unique(not_rep)) == len(not_rep)
