from abc import abstractmethod
from pathlib import Path
from typing import List, Union

from .utils.io import hdf5_load, hdf5_save_from_obj

Path_s = Union[Path, str]


class Analyzer:
    ARRAYS: List[str]
    DATAFRAMES: List[str]
    HYPERPARAMS: List[str]

    """
    Abstract class for data analysis from raw spike data in the form of `SpikeLoader` instance.
    Prevent `SpikeLoader` from being pickled along with the instance.

    All parameters required for a exact replication from an identical `SpikeLoader` instance
    and any data required for a proper functioning of helper functions should be pointed to by an instance variable.

    Any saved analysis should include proper context.

    """
    def __init__(self, **kwargs):
        if self.DATAFRAMES is not None:
            for name in self.DATAFRAMES:
                setattr(self, name, kwargs.get(name))

        if self.ARRAYS is not None:
            for name in self.ARRAYS:
                setattr(self, name, kwargs.get(name))

    @abstractmethod
    def fit(self, *args, **kwargs):
        """ Fit `X` according to params. """

    def save(self, path: Path_s, save_transformed=True, **kwargs):
        arrs = [arr for arr in self.ARRAYS if 'transformed' not in arr] if not save_transformed else self.ARRAYS
        return hdf5_save_from_obj(path, type(self).__name__, self,
                                  arrs=arrs, dfs=self.DATAFRAMES, params=self.HYPERPARAMS, **kwargs)

    def save_append(self, *args, **kwargs):
        return self.save(*args, append=True, **kwargs)

    @classmethod
    def from_hdf5(cls, path: Path_s, load_prev_run: bool = True, **kwargs):
        arrs = cls.ARRAYS if load_prev_run else None
        dfs = cls.DATAFRAMES if load_prev_run else None
        return cls(**hdf5_load(path, cls.__name__,
                               arrs=arrs, dfs=dfs, params=cls.HYPERPARAMS, **kwargs))

# if __name__ == '__main__':
#     test = SpikeLoader.from_hdf5('tests/data/raw.hdf5')
#     s = SubtractSpontAnalyzer()
#     s.fit(test.spks, test.get_idx_spont())
#     S_out = s.transform(test.S)
#     s.save('tests/data/purim.hdf5', overwrite=True)
#
#     x = SubtractSpontAnalyzer.from_hdf5('tests/data/purim.hdf5', load_prev_run=True)
#     X_out = x.transform(test.S)
