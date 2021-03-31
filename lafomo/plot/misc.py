import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import seaborn as sns


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'CMU Serif'
sns.set(font="CMU Serif")

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
    aspect = (extent[1]-extent[0])/ (extent[3]-extent[2])
    titles = [None] * len(images) if titles is None else titles
    for ax, cax, image, title in zip(grid, grid.cbar_axes, images, titles):
        im = ax.imshow(image, extent=extent, origin='lower', aspect=aspect)
        ax.grid()
        cb = plt.colorbar(im, cax=cax)
        cb.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
        if title is not None:
            ax.set_title(title)
    return grid
