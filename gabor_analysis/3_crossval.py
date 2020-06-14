import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from gabor import fit

for k in range(1, 2):
    B = pickle.loads(Path(f'field{k}.pk').read_bytes())
    for n_pca in [4, 8, 15, 30, 60, 90, 120, 180, 250]:
        print(n_pca)

        B_rz, params_jax = fit(B, n_pca)
        with open(f'gabor_{k}_{n_pca}.pk', 'wb') as f:
            pickle.dump(params_jax, f)
