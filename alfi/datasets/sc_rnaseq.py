from os import path
from pathlib import Path

import torch
import numpy as np

from .datasets import TranscriptomicTimeSeries
from scvelo import read


class Pancreas(TranscriptomicTimeSeries):
    def __init__(self, max_cells=10000, gene_index=None, cell_mask=None, data_dir='../data/', calc_moments=True):
        super().__init__()
        if gene_index is None:
            self.num_outputs = 4000
        elif type(gene_index) is tuple:
            self.num_outputs = gene_index[1] - gene_index[0]
        else:
            self.num_outputs = 2

        self.max_cells = max_cells
        self.calc_moments = calc_moments

        self.data_path = Path(data_dir)
        self.cache_path = self.data_path / 'pancreas' / 'pancreas.pt'
        if not path.exists(self.cache_path):
            self.cache_data()

        data = torch.load(self.cache_path)
        if gene_index is None:
            self.m_observed = data['m_observed']
            self.data = data['data']
        elif type(gene_index) is tuple:
            gene_index = np.arange(gene_index[0], gene_index[1])
            self.m_observed = data['m_observed'][:, [*gene_index, *(2000 + gene_index)]]
            self.data = [data['data'][i] for i in np.concatenate([gene_index, 2000+gene_index])]
        else:
            self.m_observed = data['m_observed'][:, [gene_index, 2000 + gene_index]]
            self.data = [data['data'][gene_index], data['data'][2000 + gene_index]]

        self.gene_names = data['gene_names']
        self.loom = data['loom']
        if cell_mask is not None:
            self.m_observed = self.m_observed[..., cell_mask]
            self.data[0] = self.data[0][..., cell_mask]
            self.data[1] = self.data[1][..., cell_mask]
            self.loom = self.loom[cell_mask]

    def cache_data(self):
        import scvelo as scv
        filename = self.data_path / 'pancreas' / 'endocrinogenesis_day15.h5ad'
        data = read(filename, sparse=True, cache=True)
        data.var_names_make_unique()

        scv.pp.filter_and_normalize(data, min_shared_counts=20, n_top_genes=2000)
        u = data.layers['unspliced'].toarray()[:self.max_cells]
        s = data.layers['spliced'].toarray()[:self.max_cells]
        if self.calc_moments:
            scv.pp.moments(data, n_neighbors=30, n_pcs=30)
            u = data.layers['Mu']
            s = data.layers['Ms']
        # scaling = u.std(axis=0) / s.std(axis=0)
        # u /= np.expand_dims(scaling, 0)

        loom = data
        gene_names = loom.var.index
        data = np.concatenate([s, u], axis=1)
        num_cells = data.shape[0]
        data = torch.tensor(data.swapaxes(0, 1).reshape(4000, 1, num_cells))
        m_observed = data.permute(1, 0, 2)

        data = list(data)

        torch.save({
            'data': data,
            'm_observed': m_observed,
            'gene_names': gene_names,
            'loom': loom,
        }, self.cache_path)


class SingleCellKidney(TranscriptomicTimeSeries):
    """
    scRNA-seq dataset on the human kidney.
    Accession number: GSE131685
    Parameters:
        calc_moments: bool=False whether to use the raw unspliced/spliced or moments
    """
    def __init__(self, data_dir='../data/',
                 raw_data_dir=None,
                 calc_moments=True):
        super().__init__()
        self.num_outputs = 2000
        data_path = path.join(data_dir, 'kidney/kidney1.pt')
        if path.exists(data_path):
            data = torch.load(data_path)
            self.m_observed = data['m_observed']
            self.data = data['data']
            self.gene_names = data['gene_names']
            self.loom = data['loom']
        else:
            if raw_data_dir is None:
                raise Exception('Raw data directory cannot be None if saved dataset path does not exist')
            import scvelo as scv
            kidney1 = path.join(raw_data_dir, 'kidney1.loom')
            data = scv.read_loom(kidney1)
            scv.pp.filter_and_normalize(data, min_shared_counts=20, n_top_genes=2000)
            if calc_moments:
                scv.pp.moments(data, n_neighbors=30, n_pcs=30)
                u = data.layers['Mu']
                s = data.layers['Ms']
            else:
                u = data.layers['unspliced'].toarray()
                s = data.layers['spliced'].toarray()

            self.loom = data
            self.gene_names = self.loom.var.index
            self.data = np.concatenate([u, s], axis=1)
            num_cells = self.data.shape[0]
            self.data = torch.tensor(self.data.swapaxes(0, 1).reshape(4000, 1, num_cells))
            self.m_observed = self.data.permute(1, 0, 2)

            self.data = list(self.data)
            torch.save({
                'data': self.data,
                'm_observed': self.m_observed,
                'gene_names': self.gene_names,
                'loom': self.loom,
            }, data_path)
