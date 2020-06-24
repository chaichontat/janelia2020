from __future__ import annotations

import pandas as pd

from gabor_analysis.gabor import *
from gabor_analysis.utils import *
from gabor_analysis.utils_jax import *


class GaborFit:
    def __init__(self, path, n_pc=60, f=False):
        self.path = path
        self.n_pc = n_pc
        self.f = f
        self.rf_raw = pickle.loads(Path(path).read_bytes())
        if f:
            print('Fitting Gabor.')
            self.rf_pcaed, self.params = fit(self.rf_raw, n_pc)
            self.rf_fit = make_gabor((16, 9), self.params)
            self.corr = np.mean(correlate(self.rf_fit, self.rf_pcaed))
        else:
            self.rf_pcaed, pcs = reduce_B_rank(self.rf_raw, n_pc)

    def compare_raw_rf(self, other: GaborFit):
        return {
            'mse': mse(self.rf_raw, other.rf_raw),
            'corr': correlate(self.rf_raw, other.rf_raw),
        }

    def compare_pcaed_rf(self, other: GaborFit):
        return {
            'mse': mse(self.rf_pcaed, other.rf_pcaed),
            'corr': correlate(self.rf_pcaed, other.rf_pcaed)
        }

    def compare_fit(self, other: GaborFit):
        return {
            'mse': mse(self.rf_fit, other.rf_raw),
            'corr': correlate(self.rf_fit, other.rf_raw)
        }


# %%
fold1 = GaborFit('gabor_analysis/field1.pk', f=True)
fold2 = GaborFit('gabor_analysis/field2.pk')

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
idx = onp.argwhere((cor_pc > 0.2) * (cor_fit < 0.1)).flatten()
fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(10, 6), dpi=300)
axs = onp.hstack((axs[0:2, :], axs[2:4, :])).T

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
idx_work = onp.argwhere((cor_pc > 0.2) * (cor_fit > 0.2)).flatten()
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(16, 8), dpi=200)
axs = axs.flatten()
names = 'σθλγφxy'

for i in range(7):
    sns.distplot(fold1.params[idx_work, i], ax=axs[i], label='work')
    sns.distplot(fold1.params[idx, i], ax=axs[i], label='fail')
    axs[i].legend()
    axs[i].set_title(names[i])
plt.show()
