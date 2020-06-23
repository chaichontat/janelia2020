import numpy as np
from sklearn.decomposition import PCA


def reduce_B_rank(B, n_pca):
    model = PCA(n_components=n_pca)
    B_flatten = np.reshape(B, (len(B), -1))  # n_neu x pixels.
    B_reduced = (model.fit_transform(B_flatten) @ model.components_).reshape(B.shape)
    print(np.sum(model.explained_variance_ratio_))
    pcs = model.components_.reshape((n_pca, B.shape[1], B.shape[2]))
    return B_reduced, pcs