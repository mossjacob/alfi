import numpy as np
import seaborn as sns

from abc import ABC
from matplotlib import pyplot as plt

from .colours import Colours


plt.style.use('seaborn')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'CMU Serif'
sns.set(style='white', font="CMU Serif")


class Plotter(ABC):

    def __init__(self, model, output_names, style='seaborn'):
        self.model = model
        self.output_names = output_names

        plt.style.use(style)
        sns.set(font="CMU Serif", style='white')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = 'CMU Serif'

    def plot_double_bar(self, params_mean,
                        labels=None,
                        titles=None,
                        params_var=None,
                        ground_truths=None,
                        figsize=(8.5, 3),
                        yticks=None,
                        max_plots=10):
        if labels is None:
            labels = self.output_names
        if titles is None:
            titles = self.output_names
        if ground_truths is None:
            ground_truths = np.nan * np.ones_like(params_mean)
        if params_var is None:
            params_var = [None] * len(params_mean)
        num_plots = min(max_plots, params_mean.shape[0])
        fig, axes = plt.subplots(ncols=num_plots, figsize=figsize)
        axes = [axes] if num_plots < 2 else axes
        plotnum = 0
        num_bars = params_mean.shape[-1]
        num_bars = min(max_plots, num_bars)
        for predicted, target, var, title in zip(params_mean, ground_truths, params_var, titles):
            if any(np.isnan(target)):
                axes[plotnum].bar(np.arange(num_bars), predicted[:num_bars],
                                  width=0.4,
                                  tick_label=labels[:num_bars],
                                  color=Colours.bar1_color)
            else:
                axes[plotnum].bar(np.arange(num_bars) - 0.2, predicted[:num_bars],
                                  width=0.4,
                                  tick_label=labels[:num_bars],
                                  color=Colours.bar1_color,
                                  yerr=var[:num_bars] if params_var[0] is not None else None,
                                  capsize=2)
                axes[plotnum].bar(np.arange(num_bars) + 0.2, target[:num_bars],
                                  width=0.4,
                                  color=Colours.bar2_color,
                                  align='center')

            axes[plotnum].set_title(title)
            axes[plotnum].tick_params(axis='x', labelrotation=35)
            if yticks is not None:
                axes[plotnum].set_yticks(yticks[plotnum])
            if num_bars == 1:
                axes[plotnum].set_xlim(-1, 1)
            plotnum += 1

    def plot_losses(self, trainer, last_x=50):
        plt.figure(figsize=(5, 2))
        plt.plot(np.sum(trainer.losses, axis=1)[-last_x:])
        plt.title('Total loss')
        plt.figure(figsize=(5, 2))
        plt.subplot(221)
        plt.plot(trainer.losses[-last_x:, 0])
        plt.title('Loss')
        plt.subplot(222)
        plt.plot(trainer.losses[-last_x:, 1])
        plt.title('KL-divergence')
