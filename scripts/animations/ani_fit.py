import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.image import AxesImage

from src.gabor_analysis.gabor_fit import GaborFit
from src.receptive_field.rf import ReceptiveField
from src.spikeloader import SpikeLoader

sns.set()

fps = 5

def ani(test):
    # First set up the figure, the axis, and the plot element we want to animate
    ax: Axes
    fig, ax = plt.subplots(dpi=200, figsize=(4, 4), constrained_layout=True)
    im: AxesImage = ax.imshow(test[0], cmap='twilight_shifted')
    ax.axis('off')


    # animation function.  This is called sequentially
    def animate(i):
        if i % 5 == 0:
            print(i)
        im.set_data(test[i])
        im.set_clim(u := -np.max(np.abs(test[i])), -u)
        ax.set_title(f'Neuron 11666. Step {i}.')
        return [im]

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = FuncAnimation(fig, animate, frames=30, blit=True)
    anim.save(f'test.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])

#%%
f = SpikeLoader.from_hdf5()
rf = ReceptiveField(f.img_dim)
rf.fit_neuron(f.imgs_stim, f.S)

#%%
from src.receptive_field.rf import gen_rf_rank

rf = gen_rf_rank(rf.rf_, n_pc=30)

#%%
choose = rf[11666, ...]
x = GaborFit(n_iter=500, n_pc=0, optimizer={'name': 'adam', 'step_size': 2e-2}).fit(choose[np.newaxis, ...])
y = np.array(x.params_fit).squeeze()
z = GaborFit._make_gabor((16, 9), y)

#%%
ani(z)
