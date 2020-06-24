from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from gabor_analysis.gabor_fit import GaborFit
from gabor_analysis.utils_jax import *


class GaborDiag(GaborFit):
    def compare_raw_rf(self, other: GaborDiag):
        return {
            'mse': mse(self.rf_raw, other.rf_raw),
            'corr': correlate(self.rf_raw, other.rf_raw),
        }

    def compare_pcaed_rf(self, other: GaborDiag):
        return {
            'mse': mse(self.rf_pcaed, other.rf_pcaed),
            'corr': correlate(self.rf_pcaed, other.rf_pcaed)
        }

    def compare_fit(self, other: GaborDiag):
        return {
            'mse': mse(self.rf_fit, other.rf_raw),
            'corr': correlate(self.rf_fit, other.rf_raw)
        }


if __name__ == '__main__':
    sns.set()

    fold1 = GaborDiag().fit(pickle.loads(Path('gabor_analysis/field1.pk').read_bytes()))
    fold2 = GaborDiag().pca_rf(pickle.loads(Path('gabor_analysis/field2.pk').read_bytes()))

    # %% PCA vs Gabor fit on test data.

    cor_pc = fold1.compare_pcaed_rf(fold2)['corr']
    cor_fit = fold1.compare_fit(fold2)['corr']

    sns.jointplot(cor_pc, cor_fit, kind='scatter', linewidth=0, s=10, alpha=0.1)

    df = pd.DataFrame([cor_pc, cor_fit]).T
    df.columns = ['PCA', 'Gabor Fit']
    g = sns.jointplot(data=df, x='PCA', y='Gabor Fit', kind='scatter', linewidth=0, s=5, alpha=0.1)
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('Correlations of Gabor Fit and PCA to test data.')
    g.fig.set_dpi(300)

    plt.show()

    # %% Examples of badly fitted.
    idx = np.argwhere((cor_pc > 0.2) * (cor_fit < 0.1)).flatten()
    fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(10, 6), dpi=300)
    axs = np.hstack((axs[0:2, :], axs[2:4, :])).T

    for i in range(10):
        scale = 3
        axs[i, 0].imshow(fold1.rf_pcaed[idx[i], ...], cmap='bwr', vmin=-scale, vmax=scale)
        axs[i, 1].imshow(fold1.rf_fit[idx[i], ...], cmap='bwr', vmin=-scale, vmax=scale)
        axs[i, 0].axis('off')
        axs[i, 1].axis('off')
    fig.suptitle(f'10 (cor_pc > 0.2) * (cor_gabor < 0.1) RFs')
    plt.tight_layout()
    plt.show()

    # %% Compare distribution of parameters.
    sns.set()
    idx_work = np.argwhere((cor_pc > 0.2) * (cor_fit > 0.2)).flatten()
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(16, 8), dpi=200)
    axs = axs.flatten()
    names = 'σθλγφxy'

    for i in range(7):
        sns.distplot(fold1.params[idx_work, i], ax=axs[i], label='work')
        sns.distplot(fold1.params[idx, i], ax=axs[i], label='fail')
        axs[i].legend()
        axs[i].set_title(names[i])
    plt.show()
