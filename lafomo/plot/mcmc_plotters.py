from matplotlib import pyplot as plt
import numpy as np
import arviz
from dataclasses import dataclass, field
from matplotlib.animation import FuncAnimation
import networkx as nx
import tensorflow as tf
import random

from lafomo.datasets import scaled_barenco_data


@dataclass
class PlotOptions:
    num_plot_genes: int = 20
    num_plot_tfs:   int = 20
    gene_names:     np.array = None
    tf_names:       list = None
    num_kinetic_avg:int = 50
    num_hpd:        int = 100
    true_label:     str = 'True'
    tf_present:     bool = True
    for_report:     bool = True
    ylabel:         str = ''
    protein_present:bool = True
    model_label:    str = 'Model'
    kernel_names:   list = field(default_factory=lambda:['Param 1', 'Param 2']) 


class Plotter:
    def __init__(self, data, options: PlotOptions):
        self.opt = options
        if self.opt.gene_names is None:
            self.opt.gene_names = np.array([f'Gene {j}' for j in range(data.m_obs.shape[1])])
        if self.opt.tf_names is None:
            self.opt.tf_names = np.array([f'TF {i}' for i in range(data.f_obs.shape[1])])
        self.num_tfs = data.f_obs.shape[1]
        self.num_outputs = data.m_obs.shape[1]
        self.data = data
        self.t_discretised = data.t_discretised.numpy()
        self.t = data.t_observed
        self.common_ind = data.common_indices.numpy()

    def plot_kinetics(self, results, kinetic_params, true_k=None, true_hpds=None, title='', xlabels=None):
        """
        Parameters:
            results: the results dictionary from the sampler
            kinetic_params: list of strings. The keys in `results` which are kinetic parameters of shape
                            (num_samples, num_outputs, 1)
        """
        kinetics = np.stack([results[k] for k in kinetic_params]).transpose((1, 2, 0, 3)).squeeze(-1)
        mean_kinetics = np.mean(kinetics[-self.opt.num_kinetic_avg:], axis=0)

        plot_labels = ['Initial Conditions', 'Basal rates', 'Decay rates', 'Sensitivities']
        num_x = mean_kinetics.shape[0]
        width = num_x
        height = 10
        if num_x < 2:
            width = 4
            height = 3
        plt.figure(figsize=(width, height))
        if not self.opt.for_report:
            plt.suptitle(title)

        if mean_kinetics.shape[1] < 4:
            plot_labels = plot_labels[1:]

        hpds = self.plot_bar_hpd(
            kinetics, mean_kinetics,
            xlabels,
            true_var=true_k, true_hpds=true_hpds,
            width=0.2, rotation=60)
        plt.tight_layout(h_pad=2.0)

        return mean_kinetics, hpds

    def plot_bar_hpd(self, var_samples, var, labels, true_var=None, width=0.1, titles=None, 
                     rotation=0, true_hpds=None):
        hpds = list()
        num = var.shape[0]
        plotnum = var.shape[1]*100 + 21
        for i in range(num):
            hpds.append(arviz.hdi(var_samples[-self.opt.num_hpd:, i,:], 0.95))
        hpds = np.array(hpds)
        hpds = abs(hpds - np.expand_dims(var, 2))
        for k in range(var_samples.shape[2]):
            plt.subplot(plotnum)
            plotnum += 1
            plt.bar(np.arange(num)-width, var[:, k], width=2*width, tick_label=labels, color='chocolate', label=self.opt.model_label)
            plt.errorbar(np.arange(num)-width, var[:, k], hpds[:, k].swapaxes(0,1), fmt='none', capsize=5, color='black')
            plt.xlim(-1, num)
            plt.xticks(rotation=rotation)
            if titles is not None:
                plt.title(titles[k])
            if true_var is not None:
                plt.bar(np.arange(num)+width, true_var[:, k], width=2*width, color='slategrey', align='center', label=self.opt.true_label)
                if true_hpds is not None:
                    plt.errorbar(np.arange(num)+width, true_var[:, k], true_hpds[:, k].swapaxes(0,1), fmt='none', capsize=5, color='black')
                plt.legend()
        plt.tight_layout()
        return hpds

    def plot_convergence(self, results, param_names, title=''):
        params = [results[param] for param in param_names]
        num = len(params)
        plt.figure(figsize=(8, 3 * num))
        plt.suptitle(title)

        for j in range(num):
            param = params[j]
            ax = plt.subplot(num, 1, j + 1)
            ax.set_title(param_names[j])
            plt.plot(np.squeeze(param[:, :]))  # , label=labels[v])
            plt.legend()
        plt.tight_layout()

    def plot_samples(self, samples, titles, num_samples, color='grey', scatters=None, 
                     scatter_args={}, legend=True, margined=False, sample_gap=2):
        num_components = samples[0].shape[0]
        subplot_shape = ((num_components+2)//3, 3)
        if self.opt.for_report:
            subplot_shape = ((num_components+1)//2, 2)
        if num_components <= 1:
            subplot_shape = (1, 1) 
        for j in range(num_components):
            ax = plt.subplot(subplot_shape[0], subplot_shape[1], 1+j)
            plt.title(titles[j])
            if scatters is not None:
                plt.scatter(self.t_discretised[self.common_ind], scatters[j], marker='x', label='Observed', **scatter_args)
            # plt.errorbar([n*10+n for n in range(7)], Y[j], 2*np.sqrt(Y_var[j]), fmt='none', capsize=5)

            for s in range(1, sample_gap*num_samples, sample_gap):
                kwargs = {}
                if s == 1:
                    kwargs = {'label':'Samples'}

                plt.plot(self.t_discretised, samples[-s,j,:], color=color, alpha=0.5, **kwargs)
            if j % subplot_shape[1] == 0:
                plt.ylabel(self.opt.ylabel)


            # HPD:
            bounds = arviz.hdi(samples[-self.opt.num_hpd:,j,:], 0.95)
            plt.fill_between(self.t_discretised, bounds[:, 0], bounds[:, 1], color='grey', alpha=0.3, label='95% credibility interval')

            plt.xticks(self.t)
            ax.set_xticklabels(self.t)
            if margined:
                plt.ylim(min(samples[-1, j])-2, max(samples[-1, j]) + 2)
            # if self.opt.for_report:
                # plt.ylim(-0.2, max(samples[-1, j]) + 0.2)
            plt.xlabel('Time (h)')
            if legend:
                plt.legend()
        plt.tight_layout()

    def plot_outputs(self, m_preds, replicate=0, height_mul=3, width_mul=2, indices=None):
        m_preds = m_preds[:, replicate]
        scatters = self.data.m_obs[replicate]
        if indices:
            m_preds = m_preds[:, indices]
            scatters = scatters[indices]
        height = np.ceil(m_preds.shape[1]/3)
        width = 4 if self.opt.for_report else 7
        plt.figure(figsize=(width*width_mul, height*height_mul))
        if not self.opt.for_report:
            plt.suptitle('Genes')
        names = self.opt.gene_names[indices] if indices is not None else self.opt.gene_names
        self.plot_samples(m_preds, names, self.opt.num_plot_genes, 
                          scatters=scatters, legend=not self.opt.for_report)

    def plot_latents(self, f_samples, replicate=0, scale_observed=False, plot_barenco=False, sample_gap=2):
        f_samples = f_samples[:, replicate]
        num_tfs = self.data.f_obs.shape[1]
        width = 6*min(num_tfs, 3)
        figsize=(width, 4*np.ceil(num_tfs/3))
        if self.opt.for_report and num_tfs == 1:
            figsize=(6, 4)
        plt.figure(figsize=figsize)
        scatter_args = {'s': 60, 'linewidth': 2, 'color': 'tab:blue'}
        self.plot_samples(f_samples, self.opt.tf_names, self.opt.num_plot_tfs, 
                          scatters=self.data.f_obs[replicate] if self.opt.tf_present else None,
                          scatter_args=scatter_args, margined=True, sample_gap=sample_gap)
        
        # if 'σ2_f' in model.params._fields:
        #     σ2_f = model.params.σ2_f.value
        #     plt.errorbar(t_discretised[common_indices], f_observed[0], 2*np.sqrt(σ2_f[0]),
        #                 fmt='none', capsize=5, color='blue')
        # else:
        #     σ2_f = σ2_f_pre
        # for i in range(num_tfs):
        #     f_obs = self.data.f_obs[replicate, i]
        #     if scale_observed:
        #         f_obs = f_obs / np.mean(f_obs) * np.mean(f_i)

        if plot_barenco:
            barenco_f, _ = scaled_barenco_data(np.mean(f_samples[-10:], axis=0))
            plt.scatter(self.t_discretised[self.common_ind], barenco_f, marker='x', s=60, linewidth=3, label='Barenco et al.')

        plt.tight_layout()

    def moving_average(self, a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def plot_noises(self, σ2_m_samples, σ2_f_samples):
        plt.figure(figsize=(5, 3))
        for j in range(self.opt.gene_names.shape[0]):
            ma = self.moving_average(σ2_m_samples[-1000:, j], n=20)
            plt.plot(ma, label=self.opt.gene_names[j])
        if self.opt.gene_names.shape[0] < 8:
            plt.legend()
        if σ2_f_samples is not None:
            plt.figure(figsize=(5, 3))
            for j in range(σ2_f_samples.shape[1]):
                plt.plot(σ2_f_samples[:, j])
            plt.legend()

    def plot_weights(self, weights):
        plt.figure()
        w = weights[0]
        w_0 = weights[1]
        for j in range(self.opt.gene_names.shape[0]):
            plt.plot(w[:, j], label=self.opt.gene_names[j])
        if self.opt.gene_names.shape[0] < 8:
            plt.legend()
        plt.title('Interaction weights')
        plt.figure()
        for j in range(self.opt.gene_names.shape[0]):
            plt.plot(w_0[:,j])
        plt.title('Interaction bias')

    '''
    Plots convergence with histogram on the side
    Args:
      samples: 1D array
      lims: tuple (lower, upper)
    '''
    def plot_convergence_hist(self, samples, lims=None, fig_height=6, fig_width=8, color='slategrey'):
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.005
        rect_scatter = [left, bottom, width, height]
        rect_histy = [left + width + spacing, bottom, 0.2, height]
        plt.figure(figsize=(fig_width, fig_height))

        ax_scatter = plt.axes(rect_scatter)
        ax_scatter.tick_params(direction='in', top=True, right=True)
        ax_histy = plt.axes(rect_histy)
        ax_histy.tick_params(direction='in', labelleft=False)

        # the scatter plot:
        ax_scatter.plot(samples, color=color)

        # now determine nice limits by hand:
        binwidth = 0.25
        if lims is None:
            lims = (0, np.ceil(samples.max() / binwidth) * binwidth)
        # ax_scatter.set_xlim((-lim, lim))
        ax_scatter.set_ylim(lims)
        ax_histy.hist(samples, orientation='horizontal', color=color, bins='auto')#, bins=bins)

        ax_histy.set_ylim(ax_scatter.get_ylim())

    def anim_latent(self, samples, index=0, replicate=0, interval=10):
        fig = plt.figure()
        s_min, s_max = np.min(samples[:, 0, 0, :]), np.max(samples[:, replicate, index, :])
        ax = plt.axes(xlim=(-1, 13), ylim=(s_min-0.4, s_max+0.4))
        line, = ax.plot([], [], lw=3)
        def init():
            line.set_data([], [])
            return line,
        def animate(i):
            x = np.linspace(0, 12, samples.shape[3])
            y = samples[i*interval, replicate, index]
            line.set_data(x, y)
            return line,
        anim = FuncAnimation(fig, animate, init_func=init,
                                    frames=samples.shape[0]//interval, interval=50, blit=True)
        return anim.to_html5_video()

    def plot_grn(self, results, use_basal=True, use_sensitivities=True, log=False):
        G = nx.DiGraph()
        pos=nx.spring_layout(G, seed=42)
        random.seed(42)
        np.random.seed(42)
        nodes, node_colors, sizes, edges, colors = list(), list(), list(), list(), list()

        for i in range(self.opt.tf_names.shape[0]):
            nodes.append(self.opt.tf_names[i])
            node_colors.append('slategrey')
            sizes.append(1000)

        for j in range(self.opt.gene_names.shape[0]):
            nodes.append(self.opt.gene_names[j])
            node_colors.append('chocolate')

        k = np.mean(results.k[-self.opt.num_kinetic_avg:], axis=0)
        b = k[:, 1]
        s = k[:, 3]
        w = np.mean(results.weights[0][-100:], axis=0)
        if log:
            w = np.exp(w)
        s_min = np.min(s)
        s_diff = max(s) - s_min
        b_min = np.min(b)
        b_diff = max(b) - b_min
        w_min = tf.math.reduce_min(w).numpy()
        diff = tf.math.reduce_max(w).numpy() - w_min

        for j in range(self.num_outputs):
            for i in range(self.num_tfs):
                edge = (self.opt.tf_names[i], self.opt.gene_names[j])
                weight = (s[j]-s_min) / s_diff if use_sensitivities else (w[j, i]-w_min) / diff
                colors.append(f'{1-weight}')
                edges.append(edge)
            if use_basal:
                sizes.append(int(700 + 1700 * (b[j]-b_min)/b_diff))
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        node_size = sizes if use_basal else 1000
        pos = nx.spring_layout(G, seed=42)
        pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
        nx.draw(G, pos=pos, edge_color=colors, node_color=node_colors, node_size=node_size, with_labels=True)
