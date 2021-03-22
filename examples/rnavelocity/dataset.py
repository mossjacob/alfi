from os import path

import torch
import numpy as np

from lafomo.datasets import TranscriptomicTimeSeries


class Pancreas(TranscriptomicTimeSeries):
    def __init__(self):
        super().__init__()
        import scvelo as scv
        data = scv.datasets.pancreas()
        scv.pp.filter_and_normalize(data, min_shared_counts=20, n_top_genes=2000)
        scv.pp.moments(data, n_neighbors=30, n_pcs=30)
        u = data.layers['unspliced'].toarray()[:10]
        s = data.layers['spliced'].toarray()[:10]
        print(u.shape, s.shape)

        self.num_outputs = 4000
        self.loom = data
        self.gene_names = self.loom.var.index
        self.data = np.concatenate([u, s], axis=1)
        num_cells = self.data.shape[0]
        self.data = torch.tensor(self.data.swapaxes(0, 1).reshape(4000, 1, num_cells))
        self.m_observed = self.data.permute(1, 0, 2)

        self.data = list(self.data)


class SingleCellKidney(TranscriptomicTimeSeries):
    """
    scRNA-seq dataset on the human kidney.
    Accession number: GSE131685
    Parameters:
        calc_moments: bool=False whether to use the raw unspliced/spliced or moments
    """
    def __init__(self, data_dir='./',
                 raw_data_dir=None,
                 calc_moments=True):
        super().__init__()
        self.num_outputs = 2000
        data_path = path.join(data_dir, 'kidney1.pt')
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
