# janelia2020

[![Build Status](https://travis-ci.com/chaichontat/janelia2020.svg?branch=master)](https://travis-ci.com/chaichontat/janelia2020) [![Scrutinizer Code Quality](https://scrutinizer-ci.com/g/chaichontat/janelia2020/badges/quality-score.png?b=master)](https://scrutinizer-ci.com/g/chaichontat/janelia2020/?branch=master)

Work at the Pachitariu lab during summer 2020 (aka the transition from the BC (before coronavirus) to the AD (after domestication)).

> An exercise in two-photon neural data analysis and proper scientific computing practices.

## Science

We analyzed 20,000-40,000 simulatenously imaged neurons from the mouse visual cortex. Two-photon images were deconvolved into spikes with [suite2p](https://github.com/MouseLand/suite2p).

### High-throughput Neuron Tuning

We inferred each neuron's receptive field (RF) using ridge regression and fitted a [Gabor filter](https://en.wikipedia.org/wiki/Gabor_filter) to each RF with gradient descent.

![Gabor fit parameters](https://user-images.githubusercontent.com/34997334/90060936-6f225880-dcb3-11ea-8182-0c08301eeaca.png)

### High-dimensional Representation

We identified linear subspaces in the data that encode stimuli information using canonical correlation analysis (CCA).

## DevOps

We strive to use the current best practices in [scientific computing](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005510) and [software engineering](https://www.amazon.com/Pragmatic-Programmer-journey-mastery-Anniversary/dp/0135957052) including but not limited to:

- [Testing with CI/CD](https://travis-ci.com/github/chaichontat/janelia2020)
- Extensive documentation
- [Static type checks](https://github.com/microsoft/pyright)
- [Code autoformatting](https://github.com/psf/black)
- Modular architecture
- Standardized data storage (HDF5) with parameters coupled with data
- [Perceptually uniform colormaps for visualization](https://www.kennethmoreland.com/color-advice/)
