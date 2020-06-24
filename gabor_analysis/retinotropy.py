import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from gabor_analysis.gabor_fit import GaborFit
from spikeloader import SpikeLoader

sns.set()

#%% Load Data

f = SpikeLoader()
g: GaborFit = pickle.loads(Path('gabor_analysis/gabor_30.pk').read_bytes())

#%% xy-plane vs all params.
fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(18, 10), dpi=300, constrained_layout=True)
axs = axs.flatten()
params_name = ['σ', 'θ', 'λ', 'γ', 'φ', 'pos_x', 'pos_y']
for i in range(6):
    if i != 2:
        u = axs[i].scatter(f.xpos, f.ypos, s=2 * g.corr, c=g.params[:, i], cmap='viridis', alpha=0.9)
        axs[i].grid(0)
        axs[i].set_title(params_name[i])
        fig.colorbar(u, ax=axs[i])

# Excluding x pos due to space limitations.
u = axs[2].scatter(f.xpos, f.ypos, s=2 * g.corr, c=g.params[:, -1], cmap='viridis', alpha=0.9)
axs[2].grid(0)
axs[2].set_title(params_name[-1])
fig.colorbar(u, ax=axs[2])

fig.suptitle('Params from Gabor fit of 60 PCs RF.')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#%% Hex plots vs y-pos.
def make_hexplot(var, name, title, save=None):
    df = pd.DataFrame([f.ypos, var]).T
    df.columns = ['Physical y-pos', name]
    g = sns.jointplot(data=df, x=df.columns[0], y=df.columns[1], kind='hex')
    g.fig.suptitle(title)
    g.fig.set_dpi(300)
    if save:
        plt.savefig(save)
    plt.show()

make_hexplot(g.params[:, 0], 'σ', 'Size of Gabor Fit (σ) vs y-position')
make_hexplot(g.params[:, -2], 'Gabor x-pos', 'Retinotropic x-pos of Gabor fit vs physical y-pos')
make_hexplot(g.corr, 'Gabor Fit Correlation', 'Gabor fit correlation with RF vs physical y-pos')
