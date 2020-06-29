from pathlib import Path
from typing import List

import pandas as pd
import tables


def vars_to_dict(obj, vars: List[str]):
    return {var: getattr(obj, var) for var in vars}


def hdf5_save(path, group, *, arrs: dict, dfs: dict, params: dict, overwrite=False, complib='blosc:lz4hc'):
    filters = tables.Filters(complib=complib, complevel=5)
    if not overwrite and Path(path).exists():
        raise FileExistsError('File exists and instructed to not overwrite.')

    with tables.open_file(path, 'w') as f:
        f.create_group(f.root, group)
        for k, v in arrs.items():
            f.create_carray(f'/{group}', k, obj=v, filters=filters)

        for k, v in params.items():
            f.set_node_attr(f'/{group}', k, v)

    for k, v in dfs.items():
        v.to_hdf(path, f'{group}/{k}', complib=complib)


def hdf5_save_from_obj(path, group, obj, *, arrs: List[str], dfs: List[str], params: List[str], **kwargs):
    locals_ = locals()
    converted = {name: vars_to_dict(obj, locals_[name]) for name in ['arrs', 'dfs', 'params']}
    kwargs.update(**converted)
    return hdf5_save(path, group, **kwargs)


def hdf5_load(path, group, arrs, dfs, params, skip_na=True):
    out = dict()
    with tables.open_file(path, 'r') as f:
        f.root[group]  # Check if group exists.
        for arr in arrs:
            try:
                out[arr] = f.root[group][arr].read()
            except IndexError as e:
                if not skip_na:
                    raise e
        for param in params:
            try:
                out[param] = f.get_node_attr(f'/{group}', param)
            except AttributeError as e:
                if not skip_na:
                    raise e
    for df in dfs:
        try:
            out[df] = pd.read_hdf(path, f'{group}/{df}')
        except KeyError as e:
            if not skip_na:
                raise e
    return out
