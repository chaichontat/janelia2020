import pandas as pd

from gabor import *
from utils_jax import mse, correlate, zscore_img

# %% 2-fold Cross-validation

x = [4, 8, 15, 30, 60, 90, 120, 180, 250]
mseB, mseimg, cor_pc, corimg = [], [], [], []

for k in [2, 1]:
    another = 2 if k == 1 else 1
    print(k, another)
    with open(f'field{k}.pk', 'rb') as f:
        B = zscore_img(pickle.load(f))
    with open(f'field{another}.pk', 'rb') as f:
        Bprime = zscore_img(pickle.load(f))

    for n in x:
        with open(f'gabor_{another}_{n}.pk', 'rb') as f:
            params = pickle.load(f)

        B_reduced, pcs = pca(B, n)
        Br = zscore_img(B_reduced)

        fitted = make_gabor((16, 9), params)

        mseB.append(np.mean(mse(B, fitted)))
        mseimg.append(np.mean(mse(Br, Bprime)))
        # mseBP.append(np.mean(mse(Br, fitted)))
        cor_pc.append(np.mean(correlate(B, fitted)))
        corimg.append(np.mean(correlate(Br, Bprime)))
        # corP.append(np.mean(correlate(Br, fitted)))

#%%
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12,8), dpi=300)
axs = axs.flatten()
x = [4, 8, 15, 30, 60, 90, 120, 180, 250]
for i, v in enumerate([mseB, mseimg, cor_pc, corimg]):
    axs[i].plot(x, v[:len(x)], label='Fold 1')
    axs[i].plot(x, v[len(x):], label='Fold 2')
    axs[i].legend()

axs[0].set_title('MSE between Gabor fit from Fold {legend}\n and full-PC RF from the other fold.')
axs[1].set_title('MSE between PCAed RF from Fold {legend}\n and full-PC RF from the other fold.')
axs[2].set_title('Correlation between Gabor fit from Fold {legend}\n and full-PC RF from the other fold.')
axs[3].set_title('Correlation PCAed RF from Fold {legend}\n and full-PC RF from the other fold.')

for i in [0, 1]:
    axs[i].set_ylim([1.25, 1.60])
for i in [2, 3]:
    axs[i].set_ylim([0.20, 0.40])

fig.suptitle('Error metric vs number of PCs')
plt.tight_layout(h_pad=1)
plt.show()


#%%
n = 60
k, another = 2, 1
with open(f'field{k}.pk', 'rb') as f:
    B = zscore_img(pickle.load(f))
with open(f'field{another}.pk', 'rb') as f:
    Bprime = zscore_img(pickle.load(f))
with open(f'gabor_{another}_{n}.pk', 'rb') as f:
    params = pickle.load(f)

B_reduced, pcs = pca(Bprime, n)
fitted = make_gabor((16, 9), params)

cor_pc = correlate(B_reduced, B)
cor_gabor = correlate(fitted, B)

sns.jointplot(cor_pc, cor_gabor, kind='scatter', linewidth=0, s=10, alpha=0.1)

df = pd.DataFrame([cor_pc, cor_gabor]).T
df.columns = ['PCA', 'Gabor Fit']
g = sns.jointplot(data=df, x='PCA', y='Gabor Fit', kind='scatter', linewidth=0, s=5, alpha=0.1)
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Correlations of Gabor Fit and PCA to test data.')
g.fig.set_dpi(300)

plt.show()

#%%
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

#%% Compare distribution of parameters.
idx_work = onp.argwhere((cor_pc > 0.2) * (cor_gabor > 0.2)).flatten()
fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(6, 16), dpi=300)
for i in range(5):
    sns.distplot(params[idx_work, i], ax=axs[i], label='work')
    sns.distplot(params[idx, i], ax=axs[i], label='fail')
    axs[i].legend()
plt.show()
