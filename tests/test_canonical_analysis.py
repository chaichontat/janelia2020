import numpy as np

from src.canonical_analysis.canonical_ridge import CanonicalRidge
from src.receptive_field.rf import ReceptiveField
from src.spikeloader import SpikeLoader
from src.utils.io import hdf5_load


def test_regression_canonical_ridge():
    loader = SpikeLoader.from_hdf5('tests/data/processed.hdf5')
    V1, V2 = loader.S[:, loader.pos['y'] >= 210], loader.S[:, loader.pos['y'] < 210]
    cca = CanonicalRidge().fit(V1, V2)

    V1s, V2s = cca.subtract_canon_comp(V1, V2)
    rf_v1 = ReceptiveField(loader.img_dim).fit_pc(loader.imgs_stim, V1s)
    rf_v2 = ReceptiveField(loader.img_dim).fit_pc(loader.imgs_stim, V2s)

    gnd = hdf5_load('tests/data/regression_test_data.hdf5', 'CanonicalRidge', arrs=['V1', 'V2'])
    assert np.mean((rf_v1.rf_ - gnd['V1']) / gnd['V1']) < 0.05
    assert np.mean((rf_v2.rf_ - gnd['V2']) / gnd['V2']) < 0.05
