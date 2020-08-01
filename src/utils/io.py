from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import tables
from tables.group import Group

Path_s = Union[Path, str]
To_save = Optional[Dict[str, Any]]
List_str = Optional[List[str]]


def vars_to_dict(obj: Any, vars: List[str]) -> Dict[str, Any]:
    """Get instance variables using names in `vars` from `obj`.

    Args:
        obj (Any)
        vars (List[str]): List of instance variable names.

    Returns:
        Dict[str, Any]: Instance variable name and its object.
    """
    return {var: getattr(obj, var) for var in vars}


# fmt: off
def hdf5_save(path: Path_s, group: str, *,
              arrs: To_save = None, dfs: To_save = None, params: To_save = None,
              overwrite: bool = False, append: bool = False, overwrite_group: bool = False,
              complib: str = 'blosc:lz4hc', complevel: int = 9) -> None:
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
                raise Exception('Group (current class you are trying to save) already exists in file. Consider setting `overwrite_node=True` when saving.')

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
                       **kwargs):
    locals_ = locals()
    converted = {name: vars_to_dict(obj, locals_[name])
                 for name in ['arrs', 'dfs', 'params'] if locals_[name] is not None}
    kwargs.update(**converted)
    return hdf5_save(path, group, **kwargs)


def hdf5_load(path: Path_s, group: str, *,
              arrs: List_str = None, dfs: List_str = None, params: List_str = None,
              skip_na: bool = True):
    out = dict()
    with tables.open_file(path, 'r') as f:
        f.root[group]  # Check if group exists.
        if arrs is not None:
            for arr in arrs:
                try:
                    out[arr] = f.root[group][arr].read()
                except IndexError as e:
                    if not skip_na:
                        raise e
        if params is not None:
            for param in params:
                try:
                    out[param] = f.get_node_attr(f'/{group}', param)
                except AttributeError as e:
                    if not skip_na:
                        raise e
    if dfs is not None:
        for df in dfs:
            try:
                out[df] = pd.read_hdf(path, f'{group}/{df}')
            except KeyError as e:
                if not skip_na:
                    raise e
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
