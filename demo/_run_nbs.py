import logging
from pathlib import Path

import papermill as pm

"""Execute parameterized notebooks."""

path = "data/superstim_TX57.hdf5"

parameters = dict(
    path_npz=Path(path).with_suffix(".npz").as_posix(),
    path_loader=path,
    path_rf=path,
    path_gabor=path,
)

path_output = Path("outputs/")
path_output.mkdir(parents=True, exist_ok=True)

nbs = ["preprocess", "run_rf", "retinotopy", "cca_stimuli"]

for nb in nbs:
    try:
        pm.execute_notebook(
            f"{nb}.ipynb",
            (path_output / f"{nb}_output.ipynb").as_posix(),
            parameters=parameters,
        )
    except pm.exceptions.PapermillExecutionError as e:
        logging.error(f"Error at {nb}.")
        raise e
