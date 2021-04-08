import torch
import numpy as np

from lafomo.datasets import LFMDataset


def flatten_dataset(dataset: LFMDataset):
    num_genes = dataset.num_outputs
    train_t = dataset[0][0]
    num_times = train_t.shape[0]
    m_observed = torch.stack([
        dataset[i][1] for i in range(num_genes)
    ]).view(num_genes, num_times)
    train_t = train_t.repeat(num_genes)
    train_y = m_observed.view(-1)
    return train_t, train_y


def p53_ground_truth():
    B_exact = [0.0649, 0.0069, 0.0181, 0.0033, 0.0869]
    D_exact = [0.2829, 0.3720, 0.3617, 0.8000, 0.3573]
    S_exact = [0.9075, 0.9748, 0.9785, 1.0000, 0.9680]
    return B_exact, S_exact, D_exact


def hafner_ground_truth():
    s = np.array([
        0.232461671, 0.429175332, 1.913169606, 0.569821512, 2.139812962, 0.340465324,
        4.203117214, 0.635328943, 0.920901229, 0.263968666, 1.360004451, 4.816673998,
        0.294392325, 2.281036308, 0.86918333, 2.025737447, 1.225920534, 11.39455009,
        4.229758095, 4.002484948, 32.89511304, 7.836815916])
    d = np.array([
        0.260354271, 0.253728801, 0.268641114, 0.153037374, 0.472215028, 0.185626363,
        0.210251586, 0.211915623, 0.324826082, 0.207834775, 0.322725728, 0.370265667,
        0.221598164, 0.226897275, 0.409710437, 0.398004589, 0.357308033, 0.498836353,
        0.592101838, 0.284200056, 0.399638904, 0.463468107])
    b = np.zeros_like(d)
    return b, s, d