from datetime import timedelta

import numpy as np
from hypothesis import settings

from src.gabor_analysis.gabor_fit import GaborFit
from src.utils.utils import hdf5_load

settings.register_profile('default', max_examples=10, deadline=timedelta(milliseconds=20000))
settings.load_profile('default')


def test_regression_gabor():
    rf = hdf5_load('tests/data/rf.hdf5', 'rf_gnd_truth', arrs=['neu'])['neu']
    gabor = GaborFit.from_hdf5('tests/data/gabor.hdf5')
    gabor.fit(rf)

    gnd = GaborFit.from_hdf5('tests/data/gabor.hdf5', load_prev_run=True).rf_fit
    assert np.allclose(gabor.rf_fit, gnd, atol=1e-4, rtol=1e-2)
