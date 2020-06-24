from typing import Tuple, Type

import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import train_test_split


class SpikeLoader:
    """
    Class to load suite2p npz files with stimulus information.
    Can subtract spontaneous PCs.
    """

    def __init__(self, path: str = 'superstim.npz'):
        print('Loading data.')
        self.path = path
        with np.load(self.path, mmap_mode='r') as npz:
            self._spks = np.empty(0)  # Load on demand.
            self.xpos, self.ypos = npz['xpos'], npz['ypos']
            self.frame_start, self.istim = npz['frame_start'], npz['istim']

        # Sanity checks.
        assert len(self.frame_start) == len(self.istim)  # Number of stimulations.

        self.S = zscore(self.spks[self.frame_start, :], axis=0)

    @property
    def spks(self):
        if len(self._spks) == 0:
            with np.load(self.path, mmap_mode='r') as npz:
                self._spks = npz['spks'].T  # time x neu
            assert len(self.xpos) == len(self.ypos) == self._spks.shape[1]  # Number of neurons.
        return self._spks

    def train_test_split(self, test_size: float = 0.5, random_state: int = 1256) -> Tuple:
        return train_test_split(self.S, test_size=test_size, random_state=random_state)
