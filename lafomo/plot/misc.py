import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import seaborn as sns
import pandas as pd
from seaborn import kdeplot
from .colours import Colours

plt.style.use('seaborn')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'CMU Serif'
sns.set(style='white', font="CMU Serif")


def plot_spatiotemporal_data(images, extent, nrows=1, ncols=None, titles=None):
    if ncols == None:
        ncols = len(images)
    fig = plt.figure()
    grid = ImageGrid(fig, 111,  # similar to subplot(144)
                     nrows_ncols=(nrows, ncols),
                     axes_pad=(0.8, 0.1),
                     label_mode='all',
                     share_all=False,
                     cbar_location="right",
                     cbar_mode="each",
                     cbar_size="7%",
                     cbar_pad="2%",
                 )
    aspect = (extent[1]-extent[0]) / (extent[3]-extent[2])
    titles = [None] * len(images) if titles is None else titles
    for ax, cax, image, title in zip(grid, grid.cbar_axes, images, titles):
        im = ax.imshow(image, extent=extent, origin='lower', aspect=aspect)
        cb = plt.colorbar(im, cax=cax)
        cb.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
        ax.set_xticks([np.ceil(extent[0]), np.floor(extent[1])])
        ax.set_yticks([np.ceil(extent[2]), np.floor(extent[3])])
        ax.set_xlim([extent[0], extent[1]])
        ax.set_ylim([extent[2], extent[3]])
        if title is not None:
            ax.set_title(title)
    return grid


def plot_phase(x_samples, y_samples,
               x_target=None, y_target=None,
               x_mean=None, y_mean=None, figsize=(4, 4), ax=None):
    """

    @param x_samples:
    @param y_samples:
    @param x_mean: if None, estimated from data
    @param y_mean: if None, estimated from data
    @return:
    """
    if x_mean is None:
        x_mean = x_samples.mean(0)
    if y_mean is None:
        y_mean = y_samples.mean(0)
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
    ax.plot(x_mean, y_mean, label='Prediction', color=Colours.line_color)
    if x_target is not None:
        ax.plot(x_target, y_target, label='Target', color=Colours.scatter_color)

    data_stacked = np.stack([x_samples.flatten(), y_samples.flatten()])

    ndp_df = pd.DataFrame(data_stacked.transpose(), columns=['Prey', 'Predator'])

    kdeplot(data=ndp_df, ax=ax, fill=True, x="Prey", y="Predator",
            color='pink', alpha=0.1, levels=3, thresh=.1, )
    plt.legend(loc='upper right')
