from jax import jit
from jax import numpy as jnp


def mse(data, fitted):
    x, y = check_data(data, fitted)
    return jnp.mean((x - y) ** 2, axis=1)


def mae(data, fitted):
    x, y = check_data(data, fitted)
    return jnp.mean(jnp.abs(x - y), axis=1)


@jit
def correlate(data, fitted):
    """ See Wikipedia. """
    x, y = check_data(data, fitted)

    Ex = jnp.mean(x, axis=1, keepdims=True)
    Ey = jnp.mean(y, axis=1, keepdims=True)

    cov = jnp.mean((x - Ex) * (y - Ey), axis=1)

    var_x = jnp.sqrt(jnp.var(x, axis=1))
    var_y = jnp.sqrt(jnp.var(y, axis=1))

    return cov / (var_x * var_y)


def check_data(data, fitted):
    if data.shape != fitted.shape:
        print(f'Len data={data.shape} but {fitted.shape}.')
    n = data.shape[0]
    x = jnp.reshape(data, (n, -1))
    y = jnp.reshape(fitted, (n, -1))
    return x, y


def zscore_img(img):
    ori = img.shape
    if len(img.shape) == 3:
        img = img.reshape((img.shape[0], -1))
    else:
        raise Exception('Not image!')

    # Using JAX instead of scipy.
    zed = (img - jnp.mean(img, axis=1, keepdims=1)) / jnp.sqrt(jnp.var(img, axis=1, keepdims=1))
    return zed.reshape(ori)
