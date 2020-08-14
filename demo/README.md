# Demo

## Notebooks

Notebooks are great for visualizing data but not so great for proper coding practices. See this [talk](https://www.youtube.com/watch?v=7jiPeIFXb6U). Python scripts are not really compatible with visualizations. To combine the best of both worlds, we use the following packages:

- [`Jupytext`](https://github.com/mwouts/jupytext) lets us transform `.py` files into notebooks.

- [`papermill`](https://github.com/nteract/papermill) lets us parameterize and execute notebooks.

The result is a robust system of executing Python scripts with arguments, similar to `argparse`, but with pretty notebooks as the output.

### Details

We use the [`percent`](https://jupytext.readthedocs.io/en/latest/formats.html?highlight=percent#the-percent-format) format to represent cells in Python scripts. This format is also compatible with [interactive Python](https://code.visualstudio.com/docs/python/jupyter-support-py) in Visual Studio Code.

### Run

The pipeline is implemented in `_run_nbs.py`. This script lets us run the entire analysis pipeline to a dataset.

## High-throughput Receptive Field Analysis

1. `preprocess.py`
   - Converts the raw `.npz` file into the `SpikeLoader` `HDF5` format.
   - Output basic data statistics.
   - Check for spike correlations between repeat stimuli.
2. `run_rf.py`
   - Perform ridge regression to infer the receptive field (RF) of each neuron.
   - Perform regional PCA to denoise the RFs.
3. `run_gabor.py`
   - Fit Gabor filter to each neuron using `jax`.
   - GPU usage highly recommended. This is why we split the files up in this way.
4. `retinotopy.py`
   - Visualize the results of Gabor fits.
