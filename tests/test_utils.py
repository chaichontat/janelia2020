import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import array_shapes, arrays, floating_dtypes

from src.utils.io import *


# PyTables can't deal with NULL.
@given(a1=arrays(floating_dtypes(), shape=array_shapes()), p1=st.integers(),
       p2=st.text(st.characters(blacklist_categories=("Cs",), blacklist_characters='\x00')))
def test_save_load_hdf5(a1, p1, p2):
    class ForTest:
        def __init__(self, a1, p1, p2):
            self.a1 = a1
            self.a2 = np.zeros((44,))
            self.d1 = pd.DataFrame.from_dict({'a': self.a2, 'b': self.a2})
            self.p1 = p1
            self.p2 = p2

    obj = ForTest(a1, p1, p2)
    arrs = ['a1', 'a2']
    dfs = ['d1']
    params = ['p1', 'p2']
    path = Path('tests/data/test.hdf5')

    hdf5_save_from_obj(path, 'test', obj,
                       arrs=arrs, dfs=dfs, params=params, overwrite=True)

    restored = hdf5_load('tests/data/test.hdf5', 'test', arrs=arrs, dfs=dfs, params=params)
    for arr in arrs:
        assert np.allclose(restored[arr], getattr(obj, arr), equal_nan=True)
    for df in dfs:
        assert restored[df].equals(getattr(obj, df))
    for p in params:
        assert restored[p] == getattr(obj, p)

    path.unlink()
