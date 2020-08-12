import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.image import AxesImage

from src.gabor_analysis.gabor_fit import GaborFit

sns.set()

""" Generate animation of varying parameters from GaborFit. """

fps = 30
t = 5
n = int(t * fps)


def vary(what, rang):
    p = {
        "σ": 150,
        "θ": 0.0,
        "λ": 300,
        "γ": 1,
        "φ": -0.0,
        "pos_x": 0.0,
        "pos_y": 0.0,
    }

    params = np.zeros((len(rang), len(p)))
    for i, v in enumerate(p.values()):
        params[:, i] = v
    params[:, GaborFit.KEY[what]] = rang

    return GaborFit._make_gabor((640, 640), params)


def ani(what, rang):
    test = vary(what, rang)
    # First set up the figure, the axis, and the plot element we want to animate
    ax: Axes
    fig, ax = plt.subplots(dpi=200, figsize=(4, 4), constrained_layout=True)
    im: AxesImage = ax.imshow(test[0], cmap="twilight_shifted")
    ax.axis("off")

    # animation function.  This is called sequentially
    def animate(i):
        if i % 5 == 0:
            print(i)
        im.set_data(test[i])
        im.set_clim(u := -np.max(np.abs(test[i])), -u)
        ax.set_title(f"{what} = {rang[i]: 6.2f}")
        return [im]

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = FuncAnimation(fig, animate, frames=n, blit=True)
    anim.save(f"test_{what}.mp4", fps=fps, extra_args=["-vcodec", "libx264"])


#%%
ani("σ", np.linspace(50, 200, n))

ani("λ", np.linspace(200, 400, n))
ani("γ", np.linspace(0.5, 2, n))

