import logging

from abc import abstractmethod
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from .utils.io import hdf5_load, hdf5_save_from_obj

Path_s = Union[Path, str]
T = TypeVar("T", bound="Analyzer")


class Analyzer:
    """
    Abstract class for data analysis modules.
    Saves parameters and hyperparameters in HDF5 of variables listed in class variables.

    """

    ARRAYS: List[str]
    DATAFRAMES: List[str]
    HYPERPARAMS: List[str]

    def __init__(self, load_prev_run=False, **kwargs) -> None:
        self.load_prev_run = load_prev_run
        for k, v in kwargs.items():
            if self._check_attr(k) is None and k != "path":
                raise TypeError(f"{type(self).__name__} does not take {k} as argument.")
            setattr(self, k, v)

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

        return cls(
            path=path,
            load_prev_run=load_prev_run,
            **hdf5_load(path, cls.__name__, params=cls.HYPERPARAMS, **kwargs),
        )

    def __getattr__(self, name: str) -> Any:
        if name == "path":
            raise AttributeError(f"Path not set in {type(self).__name__}.")

        if not self.load_prev_run or (res := self._check_attr(name)) is None:
            raise AttributeError

        setattr(self, name, hdf5_load(self.path, type(self).__name__, **res).get(name))
        return getattr(self, name)

    def _check_attr(self, name: str) -> Optional[Dict[str, str]]:
        if self.ARRAYS is not None and name in self.ARRAYS:
            return {"arrs": [name]}
        elif self.DATAFRAMES is not None and name in self.DATAFRAMES:
            return {"dfs": [name]}
        elif self.HYPERPARAMS is not None and name in self.HYPERPARAMS:
            return {"params": [name]}
        return None

    def __repr__(self) -> str:
        hyperparams = [f"\t{p} = {getattr(self, p)}\n" for p in self.HYPERPARAMS]
        return f"{type(self).__name__}:\n" + "".join(hyperparams)


def load_if_exists(cls: Type[Analyzer]) -> Callable:
    """Decorator to load an Analyzer object from `out_file` if it exists.
    Can be overriden with `overwrite` argument into `func`.
    Otherwise, run decorated function.

    Args:
        cls (Type[Analyzer]): Analyzer object.

    Returns:
        Callable: [description]
    """

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if kwargs.get("overwrite", False):
                logging.info(f"Overwrite in {cls.__name__}.")
                return func(*args, **kwargs)

            try:
                obj = cls.from_hdf5(kwargs["out_file"])
                logging.info(f"{cls.__name__} loaded.")
                return obj
            except (IndexError, OSError):
                # For class not in file and file not existing, respectively.
                return func(*args, **kwargs)

        return wrapper

    return decorate
