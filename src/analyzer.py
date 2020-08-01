import logging

from abc import abstractmethod
from pathlib import Path
from typing import List, Type, TypeVar, Union

import numpy as np
import pandas as pd
from .utils.io import hdf5_load, hdf5_save_from_obj

Path_s = Union[Path, str]
T = TypeVar("T", bound="Analyzer")


class Analyzer:
    """
    Abstract class for data analysis from raw spike data in the form of `SpikeLoader` instance.
    Prevent `SpikeLoader` from being pickled along with the instance.

    All parameters required for a exact replication from an identical `SpikeLoader` instance
    and any data required for a proper functioning of helper functions should be pointed to by an instance variable.

    Any saved analysis should include proper context.

    """

    ARRAYS: List[str]
    DATAFRAMES: List[str]
    HYPERPARAMS: List[str]

    def __init__(self, **kwargs) -> None:
        for name, type_ in {"ARRAYS": np.ndarray, "DATAFRAMES": pd.DataFrame}.items():
            if getattr(self, name) is not None:
                for var in getattr(self, name):
                    obj = kwargs.get(var)
                    setattr(self, var, obj)

    @abstractmethod
    def fit(self, *args, **kwargs):
        """ Fit `X` according to params. """

    def save(self, path: Path_s, save_transformed: bool = True, **kwargs) -> None:
        arrs = (
            [arr for arr in self.ARRAYS if "transformed" not in arr]
            if not save_transformed
            else self.ARRAYS
        )
        return hdf5_save_from_obj(
            path,
            group=type(self).__name__,
            obj=self,
            arrs=arrs,
            dfs=self.DATAFRAMES,
            params=self.HYPERPARAMS,
            **kwargs,
        )

    def save_append(self, *args, **kwargs) -> None:
        return self.save(*args, append=True, **kwargs)

    @classmethod
    def from_hdf5(cls: Type[T], path: Path_s, load_prev_run: bool = True, **kwargs) -> T:
        if Path(path).suffix != ".hdf5":
            logging.warning("Calling from_hdf5 but file does not have extension .hdf5.")

        arrs = cls.ARRAYS if load_prev_run else None
        dfs = cls.DATAFRAMES if load_prev_run else None
        return cls(
            **hdf5_load(
                path, cls.__name__, arrs=arrs, dfs=dfs, params=cls.HYPERPARAMS, **kwargs
            )
        )

    def __repr__(self) -> str:
        hyperparams = [f"\t{p} = {getattr(self, p)}\n" for p in self.HYPERPARAMS]
        return f"{type(self).__name__}:\n" + "".join(hyperparams)


# if __name__ == '__main__':
#     test = SpikeLoader.from_hdf5('tests/data/raw.hdf5')
#     s = SubtractSpontAnalyzer()
#     s.fit(test.spks, test.get_idx_spont())
#     S_out = s.transform(test.S)
#     s.save('tests/data/purim.hdf5', overwrite=True)
#
#     x = SubtractSpontAnalyzer.from_hdf5('tests/data/purim.hdf5', load_prev_run=True)
#     X_out = x.transform(test.S)
