import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import tables
from scipy import ndimage as ndi
from scipy.stats import zscore
from sklearn.model_selection import train_test_split

os.environ['BLOSC_NTHREADS'] = '8'


@dataclass
class SpikeLoader:
    path: str
    pos: pd.DataFrame
    istim: pd.Series
    img_scale: float

    spks: np.ndarray = np.empty(0)
    imgs: np.ndarray = np.empty(0)

    _S: np.ndarray = np.empty(0)
    _imgs_stim: np.ndarray = np.empty(0)
    _img_dim: np.ndarray = np.empty(0)

    """ Anything with an `_` prefix means that it's a product of this class and 
    it is only there to save time. Not expected for a first run. """

    @classmethod
    def from_npz(cls, path: str = 'data/superstim32.npz', img_scale: float = 0.25):
        with np.load(path, mmap_mode='r') as npz:
            pos = pd.DataFrame({'x': npz['xpos'], 'y': npz['ypos']})
            istim = pd.Series(npz['istim'], index=npz['frame_start'])
            spks = npz['spks'].T
            imgs = ndi.zoom(np.transpose(npz['img'], (2, 0, 1)), (1, img_scale, img_scale), order=1)
        del npz
        return cls(**{k: v for k, v in locals().items() if k != 'cls'})

    @property
    def S(self):
        if len(self._S) == 0:
            self._S = zscore(self.spks[self.istim.index, :], axis=0)
        return self._S

    @property
    def img_dim(self):
        if len(self._img_dim) > 0:
            assert len(self._img_dim) == 2
            return self._img_dim
        return self.imgs.shape[1:]

    @property
    def imgs_stim(self):
        if len(self._imgs_stim) == 0:
            X = np.reshape(self.imgs[self.istim, ...], [len(self.istim), -1])
            self._imgs_stim = zscore(X, axis=0) / np.sqrt(len(self.istim))
        return self._imgs_stim  # (stim x pxs)

    def get_idx_rep(self, return_onetimers=False):
        istim: np.ndarray = self.istim.array.to_numpy()
        unq, unq_cnt = np.unique(istim, return_counts=True)
        idx_firstrep = unq[np.argwhere(unq_cnt > 1)]  # idx of repeating img
        idx = np.zeros([len(idx_firstrep), np.max(unq_cnt)], dtype=istim.dtype)
        for i in range(len(idx_firstrep)):
            curr = np.where(istim == idx_firstrep[i])[0]
            idx[i, :curr.size] = curr
        if return_onetimers:
            idx_one = np.where(np.isin(np.arange(len(istim)), np.array(idx).flatten(), invert=True))[0]
            return idx, idx_one
        else:
            return idx

    def train_test_split(self, test_size: float = 0.5, random_state: int = 1256) -> Tuple:
        return train_test_split(self.imgs_stim, self.S, test_size=test_size, random_state=random_state)

    def save(self, path='data/examples.hdf5'):
        filters = tables.Filters(complib='blosc:lz4hc', complevel=5)
        folder = type(self).__name__
        with tables.open_file(path, 'w') as f:
            group = f.create_group(f.root, folder)
            f.create_carray(group, 'imgs', obj=self.imgs, filters=filters)
            f.create_carray(group, 'spks', obj=self.spks, filters=filters)
        self.istim.to_hdf(path, f'{folder}/istim')
        self.pos.to_hdf(path, f'{folder}/pos')

    def save_processed(self, path='data/processed.hdf5'):
        filters = tables.Filters(complib='blosc:lz4hc', complevel=5)
        folder = type(self).__name__
        with tables.open_file(path, 'w') as f:
            group = f.create_group(f.root, folder)
            im = f.create_carray(group, 'imgs_stim', obj=self.imgs_stim, filters=filters)
            im.attrs.img_scale = self.img_scale
            im.attrs.img_dim = self.img_dim
            f.create_carray(group, 'S', obj=self.S, filters=filters)
        self.istim.to_hdf(path, f'{folder}/istim')
        self.pos.to_hdf(path, f'{folder}/pos')

    @classmethod
    def from_hdf5(cls, path='data/processed.hdf5'):
        folder = 'SpikeLoader'
        with tables.open_file(path, 'r') as f:
            _imgs_stim = f.root.SpikeLoader.imgs_stim.read()
            img_scale = f.root.SpikeLoader.imgs_stim.attrs.img_scale
            _img_dim = f.root.SpikeLoader.imgs_stim.attrs.img_dim
            _S = f.root.SpikeLoader.S.read()
        istim = pd.read_hdf(path, f'{folder}/istim')
        pos = pd.read_hdf(path, f'{folder}/pos')
        del folder, f
        return cls(**{k: v for k, v in locals().items() if k != 'cls'})

        # with zarr.open(path, mode='w') as root:
        #     pos = root.create_group('pos')
        #     pos.create_dataset('x', data=np.array(self.pos['x']), chunks=False)
        #     pos.create_dataset('y', data=np.array(self.pos['y']), chunks=False)
        #
        #     root.create_dataset('spks', data=self.spks, chunks=(None, 1000))


if __name__ == '__main__':
    test = SpikeLoader.from_hdf5()  # from_npz()  #Loader()
    # test = Spike.from_npz()
    # test.save_processed()
