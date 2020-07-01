from abc import abstractmethod
from pathlib import Path
from typing import List, Union

from .utils.utils import hdf5_load, hdf5_save_from_obj

Path_s = Union[Path, str]


class Analyzer:
    arrs: List[str]
    dfs: List[str]
    params: List[str]

    """
    Abstract class for data analysis from raw spike data in the form of `SpikeLoader` instance.
    Prevent `SpikeLoader` from being pickled along with the instance.

    All parameters required for a exact replication from an identical `SpikeLoader` instance
    and any data required for a proper functioning of helper functions should be pointed to by an instance variable.

    Any saved analysis should include proper context.

    """
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit(self, *args, **kwargs):
        """ Fit `X` according to params. """

    def save(self, path: Path_s, save_transformed=True, **kwargs):
        arrs = [arr for arr in self.arrs if 'transformed' not in arr] if not save_transformed else self.arrs
        return hdf5_save_from_obj(path, type(self).__name__, self,
                                  arrs=arrs, dfs=self.dfs, params=self.params, **kwargs)

    def save_append(self, *args, **kwargs):
        return self.save(*args, append=True, **kwargs)

    @classmethod
    def from_hdf5(cls, path: Path_s, load_prev_run: bool = True, **kwargs):
        arrs = cls.arrs if load_prev_run else None
        dfs = cls.dfs if load_prev_run else None
        return cls(**hdf5_load(path, cls.__name__,
                               arrs=arrs, dfs=dfs, params=cls.params, **kwargs))

# if __name__ == '__main__':
#     test = SpikeLoader.from_hdf5('tests/data/raw.hdf5')
#     s = SubtractSpontAnalyzer()
#     s.fit(test.spks, test.get_idx_spont())
#     S_out = s.transform(test.S)
#     s.save('tests/data/purim.hdf5', overwrite=True)
#
#     x = SubtractSpontAnalyzer.from_hdf5('tests/data/purim.hdf5', load_prev_run=True)
#     X_out = x.transform(test.S)
