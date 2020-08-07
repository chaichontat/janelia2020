from datetime import timedelta

import numpy as np
from hypothesis import settings

from src.gabor_analysis.gabor_fit import GaborFit
from src.utils.io import hdf5_load

settings.register_profile('default', max_examples=10, deadline=timedelta(milliseconds=20000))
settings.load_profile('default')


def test_regression_gabor():
    rf = hdf5_load('tests/data/test_data.hdf5', 'ReceptiveField', arrs=['neu'])['neu']
    gabor = GaborFit.from_hdf5('tests/data/test_data.hdf5', load_prev_run=False)
    gabor.fit(rf)

    gnd = GaborFit.from_hdf5('tests/data/test_data.hdf5').rf_fit
    assert np.mean((gabor.rf_fit - gnd)) < 0.05
