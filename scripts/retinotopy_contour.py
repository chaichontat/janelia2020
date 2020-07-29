import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage.filters import gaussian_filter

from src.gabor_analysis.gabor_fit import GaborFit
from src.spikeloader import SpikeLoader


def gen_grid(var, σ):
    zi = griddata((df['x'], df['y']), df[var], (xi[None, :], yi[:, None]), method='nearest')
    return gaussian_filter(zi, σ, mode='mirror', truncate=3.)


if __name__ == '__main__':
    loader = SpikeLoader.from_hdf5('data/raw.hdf5')
    g = GaborFit.from_hdf5('data/gabor.hdf5')
    df = pd.DataFrame(data=g.params_fit, columns=list(GaborFit.KEY.keys()))
    df = df.join(loader.pos)

    xi = np.linspace(df['x'].min(), df['x'].max(), 2000)
    yi = np.linspace(df['y'].min(), df['y'].max(), 2000)


    z_x, z_y = gen_grid('pos_x', 150), gen_grid('pos_y', 100),

    #%%
    fig, ax = plt.subplots(dpi=300)
    con = ax.contour(xi, yi, z_x, cmap='magma', levels=np.arange(-10.5, -2, 1.), alpha=1.)
    ax.clabel(con, inline=True, fontsize=8)
    con = ax.contourf(xi, yi, z_y, cmap='twilight_shifted', levels=8, alpha=0.8)
    cbar = fig.colorbar(con)
    cbar.set_label('Altitude')
    ax.set_title('Retinotopy\n Contour lines indicate azimuth.')
    # plt.scatter(df['x'], df['y'], s=1, c=df['pos_y'])
    plt.show()
