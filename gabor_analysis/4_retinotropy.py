import pandas as pd

from gabor import *
from utils import *
from utils_jax import correlate

sns.set()

#%% Load Data
params = pickle.loads(Path('gabor_60.pk').read_bytes())
with np.load('superstim.npz') as npz:
    img, spks = npz['img'], npz['spks']
    xpos, ypos = npz['xpos'], npz['ypos']
    frame_start, istim = npz['frame_start'], npz['istim']

B = pickle.loads(Path('field.pk').read_bytes())
B_reduced, pcs = reduce_B_rank(B, 60)
fitted = make_gabor((16, 9), params)

corr = correlate(B, fitted)

#%% xy-plane vs all params.
fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(18, 10), dpi=300)
axs = axs.flatten()
params_name = ['σ', 'θ', 'λ', 'γ', 'φ', 'pos_x', 'pos_y']
for i in range(6):
    if i != 2:
        u = axs[i].scatter(xpos, ypos, s=2*corr, c=params[:, i], cmap='viridis', alpha=0.9)
        axs[i].grid(0)
        axs[i].set_title(params_name[i])
        fig.colorbar(u, ax=axs[i])

u = axs[2].scatter(xpos, ypos, s=2*corr, c=params[:, -1], cmap='viridis', alpha=0.9)
axs[2].grid(0)
axs[2].set_title(params_name[-1])
fig.colorbar(u, ax=axs[2])

fig.suptitle('Params from Gabor fit of 60 PCs RF.')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#%% Hex plots vs y-pos.
def make_hexplot(var, name, title, save=None):
    df = pd.DataFrame([ypos, var]).T
    df.columns = ['Physical y-pos', name]
    g = sns.jointplot(data=df, x=df.columns[0], y=df.columns[1], kind='hex')
    g.fig.suptitle(title)
    g.fig.set_dpi(300)
    if save:
        plt.savefig(save)
    plt.show()

make_hexplot(params[:, 0], 'σ', 'Size of Gabor Fit (σ) vs y-position')
make_hexplot(params[:, -2], 'Gabor x-pos', 'Retinotropic x-pos of Gabor fit vs physical y-pos')
make_hexplot(corr, 'Gabor Fit Correlation', 'Gabor fit correlation with RF vs physical y-pos')
