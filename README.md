# (In lieu of) Janelia Undergraduate Scholars Program 2020

[![GitHub Actions](https://github.com/chaichontat/janelia2020/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/chaichontat/janelia2020/actions/workflows/python-package-conda.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/chaichontat/janelia2020/tree/master/demo/outputs/)
[![DOI](https://zenodo.org/badge/272083279.svg)](https://zenodo.org/badge/latestdoi/272083279)

Work at the Pachitariu lab during summer 2020 (aka _annus coronalis_).

> An exercise in two-photon neural data analysis and proper scientific computing practices.

## Science Overview

We analyzed 20,000-40,000 simulatenously imaged neurons from the mouse visual cortex. Two-photon images were deconvolved into spikes with [suite2p](https://github.com/MouseLand/suite2p). More details [here](demo/). Check analysis demos at [nbviewer](https://nbviewer.jupyter.org/github/chaichontat/janelia2020/tree/master/demo/outputs/)!

### High-throughput Receptive Field Analysis

We inferred and characterized each neuron's receptive field (RF) using ridge regression and fitted a [Gabor filter](https://en.wikipedia.org/wiki/Gabor_filter) to each RF with gradient descent.

![Gabor fit parameters](https://user-images.githubusercontent.com/34997334/90060936-6f225880-dcb3-11ea-8182-0c08301eeaca.png)

### High-dimensional Representation in Spiking Data

We identified linear [communication subspaces](https://doi.org/10.1016/j.neuron.2019.01.026) using [regularized canonical correlation analysis](http://www2.imm.dtu.dk/pubdb/edoc/imm4981.pdf) (CCA) and explored the sensitivity of the results to the number of stimuli.

![CCA](https://user-images.githubusercontent.com/34997334/90169979-5c6c5a00-dd6d-11ea-8160-51a334965f25.png)

## DevOps

We strive to use the current best practices in [scientific computing](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005510) and [software engineering](https://www.amazon.com/Pragmatic-Programmer-journey-mastery-Anniversary/dp/0135957052) including but not limited to:

- [Testing with CI/CD](https://travis-ci.com/github/chaichontat/janelia2020)
- Extensive documentation
- [Static type checks](https://github.com/microsoft/pyright)
- [Code autoformatting](https://github.com/psf/black)
- Modular architecture
- Standardized data storage (HDF5) with parameters coupled to data
- [Perceptually uniform colormaps for visualization](https://www.kennethmoreland.com/color-advice/)
