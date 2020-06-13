import pickle

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

with open('field1.pk', 'rb') as f:
    B1 = pickle.load(f)

with open('field2.pk', 'rb') as f:
    B2 = pickle.load(f)

#%%
fig, axs = plt.subplots(nrows=1, ncols=2, dpi=300)
axs = axs.flatten()

idx = 1485
axs[0].imshow(B1[:, :, idx])
axs[1].imshow(B2[:, :, idx])
plt.show()