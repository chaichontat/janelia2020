import pandas as pd

from gabor_analysis.gabor_fit import *
from gabor_analysis.receptive_field import reduce_rf_rank
from gabor_analysis.utils_jax import *

# %% 2-fold Cross-validation
"""
Determine the optimal number of PCs for denoising spiking data.
"""

ns = [60]

# Fit Gabors
for k in range(1, 3):
    B = pickle.loads(Path(f'gabor_analysis/field{k}.pk').read_bytes())
    for n_pca in ns:
        print(n_pca)

        B_rz, params_jax = fit(B, n_pca)
        with open(f'gabor_analysis/gabor_{k}_{n_pca}.pk', 'wb') as f:
            pickle.dump(params_jax, f)

# %% Use fitted parameters to generate Gabors and compute losses.
mseB, mseimg, cor_pc, corimg = [], [], [], []
for k in [2, 1]:
    """ k is where Gabor is fit. """
    another = 2 if k == 1 else 1
    print(k, another)

    B = pickle.loads(Path(f'gabor_analysis/field{k}.pk').read_bytes())
    Bprime = pickle.loads(Path(f'gabor_analysis/field{another}.pk').read_bytes())

    for n in ns:
        params = pickle.loads(Path(f'gabor_analysis/gabor_{k}_{n}.pk').read_bytes())

        B_reduced, pcs = reduce_rf_rank(B, n)
        Br = zscore_img(B_reduced)

        fitted = make_gabor((16, 9), params)

        mseB.append(np.mean(mse(Bprime, fitted)))
        mseimg.append(np.mean(mse(Bprime, Br)))
        cor_pc.append(np.mean(correlate(Bprime, fitted)))
        corimg.append(np.mean(correlate(Bprime, Br)))

# %%
# fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12,8), dpi=300)
# axs = axs.flatten()
#
# for i, v in enumerate([mseB, mseimg, cor_pc, corimg]):
#     axs[i].plot(ns, v[:len(ns)], label='Fold 1')
#     axs[i].plot(ns, v[len(ns):], label='Fold 2')
#     axs[i].legend()
#
# axs[0].set_title('MSE between Gabor fit from Fold {legend}\n and full-PC RF from the other fold.')
# axs[1].set_title('MSE between PCAed RF from Fold {legend}\n and full-PC RF from the other fold.')
# axs[2].set_title('Correlation between Gabor fit from Fold {legend}\n and full-PC RF from the other fold.')
# axs[3].set_title('Correlation PCAed RF from Fold {legend}\n and full-PC RF from the other fold.')
#
# for i in [0, 1]:
#     axs[i].set_ylim([1.25, 1.60])
# for i in [2, 3]:
#     axs[i].set_ylim([0.20, 0.40])
#
# fig.suptitle('Error metric vs number of PCs')
# plt.tight_layout(h_pad=1)
# plt.show()


# %% PCA vs Gabor fit on test data.
n = 60
k, another = 2, 1
B = zscore_img(pickle.loads(Path(f'gabor_analysis/field{k}.pk').read_bytes()))
Bprime = zscore_img(pickle.loads(Path(f'gabor_analysis/field{another}.pk').read_bytes()))
params = pickle.loads(Path(f'gabor_analysis/gabor_{k}_{n}.pk').read_bytes())

B_reduced, pcs = reduce_rf_rank(B, n)
fitted = make_gabor((16, 9), params)

cor_pc = correlate(B_reduced, Bprime)
cor_gabor = correlate(fitted, Bprime)

sns.jointplot(cor_pc, cor_gabor, kind='scatter', linewidth=0, s=10, alpha=0.1)

df = pd.DataFrame([cor_pc, cor_gabor]).T
df.columns = ['PCA', 'Gabor Fit']
g = sns.jointplot(data=df, x='PCA', y='Gabor Fit', kind='scatter', linewidth=0, s=5, alpha=0.1)
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Correlations of Gabor Fit and PCA to test data.')
g.fig.set_dpi(300)

plt.show()

#%% Examples of badly fitted.
idx = onp.argwhere((cor_pc > 0.2) * (cor_gabor < 0.1)).flatten()
fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(10, 6), dpi=300)
axs = onp.hstack((axs[0:2, :], axs[2:4, :])).T

for i in range(10):
    scale = 3
    axs[i, 0].imshow(B_reduced[idx[i], ...], cmap='bwr', vmin=-scale, vmax=scale)
    axs[i, 1].imshow(fitted[idx[i], ...], cmap='bwr', vmin=-scale, vmax=scale)
    axs[i, 0].axis('off')
    axs[i, 1].axis('off')
fig.suptitle(f'10 (cor_pc > 0.2) * (cor_gabor < 0.1) RFs')
plt.tight_layout()
plt.show()

# %% Compare distribution of parameters.
sns.set()
idx_work = onp.argwhere((cor_pc > 0.2) * (cor_gabor > 0.2)).flatten()
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(16, 8), dpi=200)
axs = axs.flatten()
names = 'σθλγφxy'

for i in range(7):
    sns.distplot(params[idx_work, i], ax=axs[i], label='work')
    sns.distplot(params[idx, i], ax=axs[i], label='fail')
    axs[i].legend()
    axs[i].set_title(names[i])
plt.show()

# %% Show bad
i = 6
p = params[idx[:20]].copy()
print(p[i])
p[i, 1] = 0.05 * π
p[i, 2] = 0.9  # σ0 θ1 γ2 λ3
p[i, 3] = 1.1
# p[4, 3] = 0.9
scale = 3
fig, axs = plt.subplots(ncols=2, figsize=(10, 6), dpi=300)
axs[0].imshow(B_reduced[idx[i], ...], cmap='bwr', vmin=-scale, vmax=scale)
axs[1].imshow(make_gabor((16, 9), p)[i, ...], cmap='bwr', vmin=-scale, vmax=scale)
plt.show()

# %%
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(2)
gmm.fit(df)
