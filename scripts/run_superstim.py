# %%
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

from src.canonical_analysis.canonical_ridge import CanonicalRidge
from src.gabor_analysis.gabor_fit import GaborFit
from src.receptive_field.rf import ReceptiveField
from src.spikeloader import SpikeLoader, convert_x

sns.set()


def preprocess(out_file='data/processed.hdf5', in_file='data/superstim32.npz'):
    loader = SpikeLoader.from_npz(in_file, img_scale=0.25)
    x, z = convert_x(loader.pos['x'], offset=5, width=473, gap=177)
    loader.pos['x'] = x
    loader.pos['z'] = 150 + z * 25
    assert loader.pos['x'].min() == 0
    assert loader.pos['x'].max() == 473

    loader.save('data/raw.hdf5', overwrite=True)
    loader.save_processed(out_file, overwrite=True)
    return loader


def run_receptive_field(loader, out_file='data/rf.hdf5'):
    rf = ReceptiveField(loader.img_dim, lamda=1.1)
    rf.fit_neuron(loader.imgs_stim, loader.S)
    rf.save(out_file, overwrite=True)
    return rf


def run_gabor(rf, out_file='data/gabor.hdf5'):
    g = GaborFit(n_pc=30, n_iter=1000, optimizer={'name': 'adam', 'step_size': 2e-2}).fit(rf.rf_)
    g.save(out_file, overwrite=True)
    return g


def run_canonical_ridge(loader, out_file='data/cca.hdf5'):
    V1 = loader.pos['y'] > 200
    V2 = loader.pos['y'] <= 200
    cca = CanonicalRidge()
    cca.fit(loader.S[:, V1], loader.S[:, V2])
    cca.save(out_file, overwrite=True)
    return cca


def rotate_neuron_loc(loader, gabor, return_θ=False):
    # Get data and center.
    pos = loader.pos[['x', 'y']].to_numpy().astype(np.float)
    pos -= np.mean(pos, axis=0)

    # Standardize z in both dims.
    y = gabor.params_fit[:, GaborFit.KEY['pos_x']]
    xs, ys = [LinearRegression().fit(pos[:, i, None], y).coef_ for i in range(2)]
    pos /= np.array([xs, ys]).flatten()

    # Get angle from multiple linear regression.
    res = LinearRegression().fit(pos, y)
    θ = np.pi / 2 - np.arctan(res.coef_[1] / res.coef_[0])

    # Rotation matrix
    c, s = np.cos(θ), np.sin(θ)
    R = np.array(((c, -s), (s, c)))

    # Rotate data.
    pos = loader.pos[['x', 'y']].to_numpy().astype(np.float)
    pos = np.rint(pos @ R)
    df = pd.DataFrame(data=pos, columns=['x', 'y']).join(pos['z'])
    return df, θ if return_θ else df


if __name__ == '__main__':
    loader = preprocess('data/superstim32.npz')
    rf = run_receptive_field(loader)

    gabor_path = 'data/gabor.hdf5'
    g = GaborFit.from_hdf5(gabor_path) if Path(gabor_path).exists() else run_gabor(rf)

    loader.pos = rotate_neuron_loc(loader, g)
    loader.save_processed('data/processed.hdf5', overwrite=True)

    # cca = run_canonical_ridge(loader)

    #
    # V1 = loader.pos['y'] > 200
    # V2 = loader.pos['y'] <= 200
    # V1c, V2c = cca.transform(loader.S[:, V1], loader.S[:, V2])
    # V1s, V2s = cca.subtract_canon_comp(loader.S[:, V1], loader.S[:, V2])
