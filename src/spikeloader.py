import os
from dataclasses import dataclass
from typing import Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import zscore
from sklearn.model_selection import train_test_split

from .utils.io import hdf5_load, hdf5_save_from_obj


os.environ['BLOSC_NTHREADS'] = '8'
Path_str = Union[str, Path]


@dataclass
class SpikeLoader:
    path: Path_str

    pos: pd.DataFrame  # (n_neu ✕ 2)
    istim: pd.Series  # (n_stim). Index is (t).
    img_scale: float

    spks: np.ndarray = np.empty(0)  # (t ✕ n_neu)
    imgs: np.ndarray = np.empty(0)  # (n_img ✕ y ✕ x)

    _S: np.ndarray = np.empty(0)  # (n_stim ✕ n_neu)
    _imgs_stim: np.ndarray = np.empty(0)  # (n_stim ✕ n_px)
    img_dim: np.ndarray = np.empty(0)  # (y, x)

    """ Anything with an `_` prefix means that it's a product of this class and 
    it is only there to save time. Not expected for a first run. """

    @classmethod
    def from_npz(cls, path: Path_str = 'data/superstim32.npz', img_scale: float = 0.25):
        with np.load(path, mmap_mode='r') as npz:
            pos = pd.DataFrame({'x': npz['xpos'], 'y': npz['ypos']})
            istim = pd.Series(npz['istim'], index=npz['frame_start'])
            spks = npz['spks'].T
            imgs = ndi.zoom(np.transpose(npz['img'], (2, 0, 1)), (1, img_scale, img_scale), order=1)
            img_dim = imgs.shape[1:]
        del npz
        return cls(**{k: v for k, v in locals().items() if k != 'cls'})

    @property
    def S(self):
        if len(self._S) == 0:
            self._S = zscore(self.spks[self.istim.index, :], axis=0).astype(np.float32)
        return self._S

    @property
    def imgs_stim(self):
        if len(self._imgs_stim) == 0:
            X = np.reshape(self.imgs[self.istim, ...], [len(self.istim), -1])
            self._imgs_stim = (zscore(X, axis=0) / np.sqrt(len(self.istim))).astype(np.float32)
        return self._imgs_stim  # (stim x pxs)

    def get_idx_rep(self, return_onetimers=False):
        istim: np.ndarray = self.istim.array.to_numpy()
        unq, unq_cnt = np.unique(istim, return_counts=True)
        idx_firstrep = unq[np.argwhere(unq_cnt > 1)]  # idx of repeating img
        idx = -1 * np.ones([len(idx_firstrep), np.max(unq_cnt)], dtype=istim.dtype)
        for i in range(len(idx_firstrep)):
            curr = np.where(istim == idx_firstrep[i])[0]
            idx[i, :curr.size] = curr
        if return_onetimers:
            idx_one = np.where(np.isin(np.arange(len(istim)), np.array(idx).flatten(), invert=True))[0]
            return idx, idx_one
        else:
            return idx

    def get_idx_spont(self) -> np.ndarray:
        idx_spont = \
            np.where(np.isin(np.arange(np.max(self.istim.index) + 1), self.istim.index,
                             assume_unique=True, invert=True))[0]  # Invert indices.
        assert idx_spont.size + self.istim.index.size == self.spks.shape[0]
        return idx_spont

    def train_test_split(self, test_size: float = 0.5, random_state: int = 1256) -> Tuple:
        return train_test_split(self.imgs_stim, self.S, test_size=test_size, random_state=random_state)

    def save(self, path: Path_str = 'data/examples.hdf5', overwrite=False):
        arrs = ['imgs', 'spks']
        dfs = ['istim', 'pos']
        params = ['img_scale', 'img_dim']
        return hdf5_save_from_obj(path, 'SpikeLoader', self, arrs=arrs, dfs=dfs, params=params, overwrite=overwrite)

    def save_processed(self, path: Path_str = 'data/processed.hdf5', overwrite=False):
        self.imgs_stim  # Process everything.
        self.S
        arrs = ['_imgs_stim', '_S']
        dfs = ['istim', 'pos']
        params = ['img_scale', 'img_dim']
        return hdf5_save_from_obj(path, 'SpikeLoader', self, arrs=arrs, dfs=dfs, params=params, overwrite=overwrite)

    @classmethod
    def from_hdf5(cls, path: Path_str = 'data/processed.hdf5'):
        arrs = ['imgs', 'spks', '_imgs_stim', '_S']
        dfs = ['istim', 'pos']
        params = ['img_scale', 'img_dim']
        return cls(path=path, **hdf5_load(path, 'SpikeLoader', arrs=arrs, dfs=dfs, params=params))

        # with zarr.open(path, mode='w') as root:
        #     pos = root.create_group('pos')
        #     pos.create_dataset('x', data=np.array(self.pos['x']), chunks=False)
        #     pos.create_dataset('y', data=np.array(self.pos['y']), chunks=False)
        #
        #     root.create_dataset('spks', data=self.spks, chunks=(None, 1000))


def convert_x(xpos, offset=5, width=473, gap=177):  # Total 650.
    x = xpos.copy()
    x -= offset
    z = x // (width + gap)
    x = x % (width + gap)
    return x, z



if __name__ == '__main__':
    test = SpikeLoader.from_npz()
