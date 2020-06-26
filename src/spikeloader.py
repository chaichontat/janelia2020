from functools import lru_cache
from typing import Tuple

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import zscore
from sklearn.model_selection import train_test_split


class SpikeLoader:
    """
    Class to load suite2p npz files with stimulus information.
    Primary 'raw' data storage and pre-processing before analysis.
    """

    def __init__(self, path: str = 'superstim32.npz', img_scale: float = 0.25):
        print('Loading data.')
        self.path = path
        self.img_scale = img_scale

        with np.load(self.path, mmap_mode='r') as npz:
            self.xpos, self.ypos = npz['xpos'], npz['ypos']
            self.frame_start, self.istim = npz['frame_start'], npz['istim']

        # Sanity checks.
        assert len(self.frame_start) == len(self.istim)  # Number of stimulations.
        assert len(self.xpos) == len(self.ypos)

    @property
    @lru_cache
    def S(self):
        return zscore(self.spks[self.frame_start, :], axis=0)

    @property
    @lru_cache
    def spks(self):
        with np.load(self.path, mmap_mode='r') as npz:
            return npz['spks'].T  # time x neu

    @property
    @lru_cache
    def img(self):
        with np.load(self.path, mmap_mode='r') as npz:
            return ndi.zoom(np.transpose(npz['img'], (2, 0, 1)), (1, self.img_scale, self.img_scale), order=1)

    @property
    @lru_cache
    def X(self):
        X = np.reshape(self.img[self.istim, ...], [len(self.istim), -1])
        return zscore(X, axis=0) / np.sqrt(len(self.istim))  # (stim x pxs)

    def train_test_split(self, test_size: float = 0.5, random_state: int = 1256) -> Tuple:
        return train_test_split(self.X, self.S, test_size=test_size, random_state=random_state)


# class SpikeLoader(SpikeLoader):
#     def __init__(self, *args, img_scale: float = 0.25, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.img_scale = img_scale
#
#         with np.load(self.path, mmap_mode='r') as npz:
#             self.img = np.transpose(npz['img'], (2, 0, 1))  # stim x y x x
#
#         self.img = ndi.zoom(self.img, (1, self.img_scale, self.img_scale), order=1)
#
#         # Normalized.
#         self.X = np.reshape(self.img[self.istim, ...], [len(self.istim), -1])
#         self.X = zscore(self.X, axis=0) / np.sqrt(len(self.istim))  # (stim x pxs)
#
#     def train_test_split(self, test_size: float = 0.5, random_state: int = 1256) -> Tuple:
#         return train_test_split(self.X, self.S, test_size=test_size, random_state=random_state)


if __name__ == '__main__':
    test = SpikeLoader()
