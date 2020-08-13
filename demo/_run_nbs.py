from pathlib import Path

import papermill as pm

"""Execute parameterized notebooks."""

parameters = dict(
    path_npz="data/superstim_TX57.npz",
    path_loader="data/superstim_TX57.hdf5",
    path_rf="data/superstim_TX57.hdf5",
    path_gabor="data/superstim_TX57_gabor.hdf5",
)

path_output = Path("outputs/")
path_output.mkdir(parents=True, exist_ok=True)

nbs = ["preprocess", "run_rf", "run_gabor", "retinotopy", "cca_stimuli"]

for nb in nbs:
    pm.execute_notebook(
        f"{nb}.ipynb", path_output / f"{nb}_output.ipynb", parameters=parameters
    )
