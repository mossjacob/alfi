import csv
import pandas as pd
import numpy as np
from sklearn import preprocessing
from os import path


class DataHolder(object):
    def __init__(self, data, noise, time):
        self.m_obs, self.f_obs = data
        if noise is not None:
            self.σ2_m_pre, self.σ2_f_pre = noise
        self.t_observed = time[0]
        self.t_discretised = time[1]
        self.common_indices = time[2]


def load_covid():
    with open('data/covidrld.csv', 'r', 1) as f:
        contents = f.buffer
        df = pd.read_table(contents, sep=',', index_col=0)
    replicates = 2
    columns = df.columns
    #ENSMPUG is the prefix for Ferret
    known_target_genes = [
        # 'ENSMPUG00000000676', #EDN1 
        'ENSMPUG00000002543', #OAS1
        'ENSMPUG00000013592', #IL6
        # 'ENSMPUG00000008689', #CXCL10
        'ENSMPUG00000008778', #Mx-1
        'ENSMPUG00000013925', #ISG15
        'ENSMPUG00000015816', #ISG20
        # 'ENSMPUG00000004319', #RIG-I
    ]
    known_tfs = [
        # 'ENSMPUG00000010522', #IRF1
        # 'ENSMPUG00000008845', #IRF7
        'ENSMPUG00000012530', #STAT1
        'ENSMPUG00000001431', #STAT2
    ]

    genes_df = df[df.index.isin(known_target_genes)][columns]
    tfs_df = df[df.index.isin(known_tfs)][columns]

    #Normalise across time points
    normalised = preprocessing.normalize(np.r_[genes_df.values,tfs_df.values])
    genes = normalised[:genes_df.shape[0]]
    tfs = normalised[genes_df.shape[0]:]

    genes = np.stack([genes[:, [0,2,4,6]], genes[:, [1,3,5,7]]])
    tfs = np.stack([tfs[:, [0,2,4,6]], tfs[:, [1,3,5,7]]])

    return (genes_df, np.float64(genes)), (tfs_df, np.float64(tfs)), np.array([0, 3, 7, 14])

def load_3day_dros():
    with open('data/3day/GSE47999_Normalized_Counts.txt', 'r', 1) as f:
        contents = f.buffer
        df = pd.read_table(contents, sep='\t', index_col=0)
    replicates = 3
    columns = df.columns[df.columns.str.startswith('20,000')][::replicates]
    known_target_genes = ['FBgn0011774', 'FBgn0030189', 'FBgn0031713', 'FBgn0032393', 'FBgn0037020', 'FBgn0051864']
    tf_names = ['FBgn0039044']
    genes_df = df[df.index.isin(known_target_genes)][columns]
    tfs_df = df[df.index.isin(tf_names)][columns]

    #Normalise across time points
    normalised = preprocessing.normalize(np.r_[genes_df.values,tfs_df.values])
    genes = normalised[:-1]
    tfs = np.atleast_2d(normalised[-1])
    return (genes_df, np.float64(genes)), (tfs_df, np.float64(tfs)), np.array([2, 10, 20])


def load_barenco_puma(dir_path):
    mmgmos_processed = True
    if mmgmos_processed:
        with open(path.join(dir_path, 'barencoPUMA_exprs.csv'), 'r') as f:
            df = pd.read_csv(f, index_col=0)
        with open(path.join(dir_path, 'barencoPUMA_se.csv'), 'r') as f:
            dfe = pd.read_csv(f, index_col=0)
        columns = [f'cARP{r}-{t}hrs.CEL' for r in range(1, 4) for t in np.arange(7)*2]
    else:
        with open(path.join(dir_path, 'barenco_processed.tsv'), 'r') as f:
            df = pd.read_csv(f, delimiter='\t', index_col=0)

        columns = [f'H_ARP1-{t}h.3' for t in np.arange(7)*2]

    known_target_genes = ['203409_at', '202284_s_at', '218346_s_at', '205780_at', '209295_at', '211300_s_at']
    genes = df[df.index.isin(known_target_genes)][columns]
    genes_se = dfe[dfe.index.isin(known_target_genes)][columns]

    assert df[df.duplicated()].size == 0

    index = {
        '203409_at': 'DDB2',
        '202284_s_at': 'p21',
        '218346_s_at': 'SESN1',
        '205780_at': 'BIK',
        '209295_at': 'DR5',
        '211300_s_at': 'p53'
    }
    genes.rename(index=index, inplace=True)
    genes_se.rename(index=index, inplace=True)

    # Reorder genes
    genes_df = genes.reindex(['DDB2', 'BIK', 'DR5', 'p21', 'SESN1', 'p53'])
    genes_se = genes_se.reindex(['DDB2', 'BIK', 'DR5', 'p21', 'SESN1', 'p53'])

    tfs_df = genes_df.iloc[-1:]
    genes_df = genes_df.iloc[:-1]
    genes = genes_df.values
    tfs = tfs_df.values

    tf_var = genes_se.iloc[-1:].values**2
    gene_var = genes_se.iloc[:-1].values**2

    tfs_full = np.exp(tfs + tf_var/2)
    genes_full = np.exp(genes+gene_var/2)

    tf_var_full = (np.exp(tf_var)-1)*np.exp(2*tfs + tf_var)
    gene_var_full = (np.exp(gene_var)-1)*np.exp(2*genes + gene_var) # This mistake is in Lawrence et al.

    tf_scale = np.sqrt(np.var(tfs_full[:, :7], ddof=1))
    tf_scale = np.c_[[tf_scale for _ in range(7*3)]].T
    tfs = np.float64(tfs_full / tf_scale).reshape((3, 1, 7))
    tf_var = (tf_var_full / tf_scale**2).reshape((3, 1, 7))

    gene_scale = np.sqrt(np.var(genes_full[:,:7], axis=1, ddof=1))
    gene_scale = np.c_[[gene_scale for _ in range(7*3)]].T
    genes = np.float64(genes_full / gene_scale).reshape((5, 3, 7)).swapaxes(0, 1)
    gene_var = np.float64(gene_var_full / gene_scale**2).reshape((5, 3, 7)).swapaxes(0, 1)

    return (genes_df, genes), (tfs_df, np.float64(tfs)), gene_var, tf_var, np.arange(7)*2           # Observation times


def scaled_barenco_data(f):
    scale_pred = np.sqrt(np.var(f))
    barencof = np.array([[0.0, 200.52011, 355.5216125, 205.7574913, 135.0911372, 145.1080997, 130.7046969],
                         [0.0, 184.0994134, 308.47592, 232.1775328, 153.6595161, 85.7272235, 168.0910562],
                         [0.0, 230.2262511, 337.5994811, 276.941654, 164.5044287, 127.8653452, 173.6112139]])

    barencof = barencof[0]/(np.sqrt(np.var(barencof[0])))*scale_pred
    # measured_p53 = df[df.index.isin(['211300_s_at', '201746_at'])]
    # measured_p53 = measured_p53.mean(0)
    # measured_p53 = measured_p53*scale_pred
    measured_p53 = 0
    
    return barencof, measured_p53


# From barenco paper
def barenco_params():
    return np.stack([
        np.array([2.6, 1.5, 0.5, 0.2, 1.35])[[0, 4, 2, 3, 1]],
        (np.array([1.2, 1.6, 1.75, 3.2, 2.3])*0.8/3.2)[[0, 4, 2, 3, 1]],
        (np.array([3, 0.8, 0.7, 1.8, 0.7])/1.8)[[0, 4, 2, 3, 1]]
    ]).T

