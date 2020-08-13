# Demo

## Notebooks

Notebooks are great for visualizing data but not so great for proper coding practices. See this [talk](https://www.youtube.com/watch?v=7jiPeIFXb6U). Python scripts are not really compatible with visualizations.

[`Jupytext`](https://github.com/mwouts/jupytext) allows us to execute and transform `.py` files into notebooks. To run each notebook, run the following command:

```script
jupytext --to notebook --execute notebook.py
```

Note that Visual Studio Code also allows interactive [cell-by-cell execution](https://code.visualstudio.com/docs/python/jupyter-support-py).

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
