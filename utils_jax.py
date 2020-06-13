from jax import numpy as np


def mse(data, fitted):
    x, y = check_data(data, fitted)
    return np.mean((x - y)**2, axis=1)


def correlate(data, fitted):
    """ See Wikipedia. """
    x, y = check_data(data, fitted)

    Ex = np.mean(x, axis=1, keepdims=True)
    Ey = np.mean(y, axis=1, keepdims=True)

    cov = np.mean((x - Ex) * (y - Ey), axis=1)

    var_x = np.sqrt(np.var(x, axis=1))
    var_y = np.sqrt(np.var(y, axis=1))

    return cov / (var_x * var_y)


def check_data(data, fitted):
    if len(data) != len(fitted):
        print(f'Len data={len(data)} but {len(fitted)}.')
    n = min(len(data), len(fitted))
    x = np.reshape(data[:n, ...], (n, -1))
    y = np.reshape(fitted[:n, ...], (n, -1))
    return x, y


def zscore_img(img):
    ori = img.shape
    if len(img.shape) == 3:
        img = img.reshape((img.shape[0], -1))
    else:
        raise Exception('Not image!')

    zed = (img - np.mean(img, axis=1, keepdims=1)) / np.sqrt(np.var(img, axis=1, keepdims=1))
    return zed.reshape(ori)

