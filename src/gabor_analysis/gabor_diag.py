from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .gabor_fit import GaborFit
from .utils_jax import mse, correlate
from ..receptive_field.rf import ReceptiveField, gen_rf_rank
from ..spikeloader import SpikeLoader


class GaborDiag(GaborFit):
    def compare_pcaed_rf(self, other: GaborDiag):
        return {
            'mse': mse(self.rf_pcaed, other.rf_pcaed),
            'corr': correlate(self.rf_pcaed, other.rf_pcaed)
        }

    def compare_fit(self, other: GaborDiag):
        return {
            'mse': mse(self.rf_fit, other.rf_pcaed),
            'corr': correlate(self.rf_fit, other.rf_pcaed)
        }


if __name__ == '__main__':
    loader = SpikeLoader.from_hdf5('data/processed.hdf5')
    X_train, X_test, S_train, S_test = loader.train_test_split()

    sns.set()
    rf1 = ReceptiveField(loader.img_dim)
    rf1.fit_neuron(X_train, S_train)
    fold1 = GaborDiag(n_pc=30, n_iter=1000, optimizer={'name': 'adam', 'step_size': 2e-2}).fit(rf1.rf_)

    rf2 = ReceptiveField(loader.img_dim)
    rf2.fit_neuron(X_test, S_test)
    fold2 = GaborDiag(n_pc=30, n_iter=1000, optimizer={'name': 'adam', 'step_size': 2e-2})
    fold2.rf_pcaed = gen_rf_rank(rf2.rf_, n_pc=30)

    # %% PCA vs Gabor fit on test data.
    cor_pc = fold1.compare_pcaed_rf(fold2)['corr']
    cor_fit = fold1.compare_fit(fold2)['corr']

    df = pd.DataFrame([cor_pc, cor_fit]).T
    df.columns = ['PCA Fit', 'Gabor Fit']
    with sns.color_palette('colorblind', desat=0.95):
        g = sns.jointplot(data=df, x='PCA Fit', y='Gabor Fit', kind='scatter', linewidth=0, s=5, alpha=0.1)
        plt.subplots_adjust(top=0.9)
        # g.fig.suptitle('Correlations of Gabor Fit and PCA Fit to PCAed test data.')
        g.fig.set_dpi(300)

    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # %% Examples of badly fitted.
    idx = np.argwhere((cor_pc > 0.2) * (cor_fit < 0.1)).flatten()
    fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(10, 6), dpi=300)
    axs = np.hstack((axs[0:2, :], axs[2:4, :])).T

    for i in range(10):
        scale = 3
        axs[i, 0].imshow(fold1.rf_pcaed[idx[i], ...], cmap='twilight_shifted', vmin=-scale, vmax=scale)
        axs[i, 1].imshow(fold1.rf_fit[idx[i], ...], cmap='twilight_shifted', vmin=-scale, vmax=scale)
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
        sns.distplot(fold1.params_fit[idx_work, i], ax=axs[i], label='work')
        sns.distplot(fold1.params_fit[idx, i], ax=axs[i], label='fail')
        axs[i].legend()
        axs[i].set_title(names[i])
    plt.show()
