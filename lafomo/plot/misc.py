import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'CMU Serif'
sns.set(font="CMU Serif")


def plot_before_after(before, after, extent):
    fig, axes = plt.subplots(ncols=4, gridspec_kw={'width_ratios': [14, 1, 14, 1]})
    im = axes[0].imshow(before, extent=extent, origin='lower')
    fig.colorbar(im, cax=axes[1])
    im = axes[2].imshow(after, extent=extent, origin='lower');
    fig.colorbar(im, cax=axes[3])
    axes[0].axis('off')
    axes[2].axis('off')
    return axes
