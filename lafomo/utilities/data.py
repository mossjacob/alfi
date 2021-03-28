import torch

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
