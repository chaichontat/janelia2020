from pathlib import Path

import papermill as pm

names = [
    Path('receptive_field.ipynb'),
    Path('retinotopy.ipynb')
]


for name in names:
    assert name.suffix == '.ipynb'
    pm.execute_notebook(name.as_posix(), (name.parent / f'{name.stem}_output.ipynb').as_posix())
