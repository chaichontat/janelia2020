from datetime import timedelta
from functools import partial

import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from hypothesis import settings

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
