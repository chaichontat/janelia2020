import hashlib
import logging
from functools import partial
from os import cpu_count
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import tables

Path_s = Union[Path, str]
To_save = Optional[Dict[str, Any]]
List_str = Optional[List[str]]

tables.set_blosc_max_threads(cpu_count() // 2)


def vars_to_dict(obj: Any, vars_: List[str]) -> Dict[str, Any]:
    """Get instance variables using names in `vars` from `obj`.

    Args:
        obj (Any)
        vars (List[str]): List of instance variable names.

    Returns:
        Dict[str, Any]: Instance variable name and its object.
    """
    out = dict()
    for var in vars_:
        try:
            out[var] = getattr(obj, var)
        except AttributeError:
            logging.warning(f"Attribute {var} not found when saving {type(obj).__name__}.")
    return out


# fmt: off
def hdf5_save(path: Path_s, group: str, *,
              arrs: To_save = None, dfs: To_save = None, params: To_save = None,
              overwrite: bool = False, append: bool = False, overwrite_group: bool = False,
              complib: str = 'blosc:lz4', complevel: int = 5) -> None:
# fmt: on

    filters = tables.Filters(complib=complib, complevel=complevel)

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    if append:
        mode = 'a'
        if not Path(path).exists():
            print('Append but file does not exist, saving anyway.')
    else:
        mode = 'w'
        if not overwrite and Path(path).exists():
            raise FileExistsError('File exists. Consider setting `overwrite=True`')

    with tables.open_file(path, mode) as f:
        try:
            f.create_group(f.root, group)
        except tables.exceptions.NodeError:  # Group exists.
            if overwrite_group:
                f.remove_node(f.root, group, recursive=True)
                f.create_group(f.root, group)
            else:
                raise Exception('Group (current class you are trying to save) already exists in file.'
                                'Consider setting `overwrite_node=True` when saving.')

        if arrs is not None:
            for k, v in arrs.items():
                if v is not None:
                    f.create_carray(f'/{group}', k, obj=np.asarray(v), filters=filters)

        if params is not None:
            for k, v in params.items():
                f.set_node_attr(f'/{group}', k, v)

    if dfs is not None:
        for k, v in dfs.items():
            v.to_hdf(path, f'{group}/{k}', complib=complib)


def hdf5_save_from_obj(path: Path_s, group: str, obj, *,
                       arrs: List_str = None, dfs: List_str = None, params: List_str = None,
                       **kwargs) -> None:
    locals_ = locals()
    converted = {name: vars_to_dict(obj, locals_[name])
                 for name in ['arrs', 'dfs', 'params'] if locals_[name] is not None}
    kwargs.update(**converted)
    return hdf5_save(path, group, **kwargs)


def _hdf5_get(names: List[str], func: Callable[[str, str], Any], err: BaseException,
              group: str, skip_na: bool) -> Dict[str, Any]:
    """Retrieve objects from the HDF5 file.
    
    Required as different types require different retrieval functions and throw
    different exceptions.

    Args:
        names (List[str]): List of variable names.
        func (Callable[[str, str], Any]): Function used to retrieve data.
        
        err (BaseException): Exception to catch when variable does not exist.
        group (str): Name of group/node in the HDF5 file.
        skip_na (bool): Whether to ignore variables that do not exist.

    Raises:
        e: {err} when not {skip_na}

    Returns:
        Dict[str, Any]: Dict with retrieved objects.
    """
    assert not isinstance(names, str)
    objs = dict()
    if names is not None:
        for name in names:
            try:
                objs[name] = func(group, name)
            except err as e:
                logging.warning(f"{name} not in {group}.")
                if not skip_na:
                    raise e
    return objs
    
    
def hdf5_load(path: Path_s, group: str, *,
              arrs: List_str = None, dfs: List_str = None, params: List_str = None,
              skip_na: bool = True) -> Dict[str, Any]:
    out = dict()
    get_func = partial(_hdf5_get, group=group, skip_na=skip_na)
    
    with tables.open_file(path, 'r') as f:
        f.root[group]  # Check if group exists.
        arrs_call = (arrs, lambda group, name: f.root[group][name].read(), IndexError)
        params_call = (params, lambda group, name: f.get_node_attr(f'/{group}', name), AttributeError)
        
        out.update(get_func(*arrs_call))
        out.update(get_func(*params_call))

    dfs_call = (dfs, lambda group, name: pd.read_hdf(path, f'{group}/{name}'), KeyError)
    out.update(get_func(*dfs_call))
    return out


def hdf5_list_groups(path: Path_s) -> List[str]:
    """Get the name of saved groups in file.

    Args:
        path (Path_s): HDF5 path.

    Returns:
        List[str]: List of group names.
    """
    with tables.open_file(path, 'r') as f:
        return [group._v_name for group in f.walk_groups()][1:]  # Remove root.


def sha256(path: Path_s) -> str:
    """Generate SHA-256 hash of file.

    Args:
        path (Path_s)

    Returns:
        str: Hash
    """

    h  = hashlib.sha256()
    b  = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(path, 'rb', buffering=0) as f:
        for n in iter(lambda : f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()
