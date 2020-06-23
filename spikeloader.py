from typing import Tuple

import numpy as np
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


class SpikeLoader:
    """
    Class to load suite2p npz files with stimulus information.
    Can subtract spontaneous PCs.
    """

    def __init__(self, path: str = 'superstim.npz'):
        print('Loading data.')
        self.path = path
        with np.load(path) as npz:
            self.spks = npz['spks'].T  # time x neu
            self.xpos, self.ypos = npz['xpos'], npz['ypos']
            self.frame_start, self.istim = npz['frame_start'], npz['istim']

        # Sanity checks.
        assert len(self.frame_start) == len(self.istim)  # Number of stimulations.
        assert len(self.xpos) == len(self.ypos) == self.spks.shape[1]  # Number of neurons.

        self.S = zscore(self.spks[self.frame_start, :], axis=0)

    def train_test_split(self, test_size: float = 0.5, random_state: int = 1256) -> Tuple:
        return train_test_split(self.S, test_size=test_size, random_state=random_state)
