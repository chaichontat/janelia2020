from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import tables


def vars_to_dict(obj, vars: List[str]):
    return {var: getattr(obj, var) for var in vars}


def hdf5_save(path, group, *, arrs: dict = None, dfs: dict = None, params: dict = None,
              overwrite=False, append=False, overwrite_node=False, complib='blosc:lz4hc', complevel=9):
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
        except tables.exceptions.NodeError as e:  # Node exists.
            if overwrite_node:
                f.remove_node(f.root, group, recursive=True)
                f.create_group(f.root, group)
            else:
                raise Exception('Node already exists. Consider setting `overwrite_node=True` when saving.')

        if arrs is not None:
            for k, v in arrs.items():
                f.create_carray(f'/{group}', k, obj=np.asarray(v), filters=filters)

        if params is not None:
            for k, v in params.items():
                f.set_node_attr(f'/{group}', k, v)

    if dfs is not None:
        for k, v in dfs.items():
            v.to_hdf(path, f'{group}/{k}', complib=complib)


def hdf5_save_from_obj(path, group, obj, *,
                       arrs: List[str] = None, dfs: List[str] = None, params: List[str] = None, **kwargs):
    locals_ = locals()
    converted = {name: vars_to_dict(obj, locals_[name])
                 for name in ['arrs', 'dfs', 'params'] if locals_[name] is not None}
    kwargs.update(**converted)
    return hdf5_save(path, group, **kwargs)


def hdf5_load(path, group, arrs=None, dfs=None, params=None, skip_na=True):
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
