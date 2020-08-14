import logging
from pathlib import Path

import nbconvert
import jupytext
import papermill as pm

"""Execute parameterized notebooks.

1. Convert py to ipynb with nbconvert and set kernelspec.
2. Run said notebook with papermill.
3. Delete said notebook.

"""

data_path = "data/superstim_TX57.hdf5"
nbs = ["preprocess", "run_rf", "retinotopy", "cca_stimuli"]
path_output = Path("outputs/")

parameters = dict(
    path_npz=Path(data_path).with_suffix(".npz").as_posix(),
    path_loader=data_path,
    path_rf=data_path,
    path_gabor=data_path,
)

path_output.mkdir(parents=True, exist_ok=True)

for nb in nbs:
    try:
        ntb = jupytext.read(nb + ".py")
        ntb.metadata["kernelspec"] = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
        jupytext.write(ntb, nb + ".ipynb")
        
        print(nb)
        pm.execute_notebook(
            f"{nb}.ipynb",
            output_path=(path_output / f"{nb}_output.ipynb").as_posix(),
            parameters=parameters,
        )
        
        
    except pm.exceptions.PapermillExecutionError as e:
        logging.error(f"Error at {nb}.")
        raise e
    finally:
        Path(nb + ".ipynb").unlink(missing_ok=True)
