# Notebooks

Notebooks are great for visualizing data but not so great for proper coding practices. See this [talk](https://www.youtube.com/watch?v=7jiPeIFXb6U). Python scripts are not really compatible with visualizations.

[`Jupytext`](https://github.com/mwouts/jupytext) allows us to execute and transform `.py` files into notebooks. To run each notebook, run the following command:

```script
jupytext --to notebook --execute notebook.py
```

Note that Visual Studio Code also allows interactive [cell-by-cell execution](https://code.visualstudio.com/docs/python/jupyter-support-py).
