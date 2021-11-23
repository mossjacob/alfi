import torch
import numpy as np
import pandas as pd
from pathlib import Path

from alfi.datasets import HomogeneousReactionDiffusion, ReactionDiffusionGenerator

"""
This script generates a synthetic reaction diffusion dataset .
"""


def save_dataset(toydata):
    """
    data_dir: the directory where the toy data and intermediate data lies. Will also be saved here.
    """
    data_dir = '../data'
    temp = pd.read_csv(Path(data_dir) / 'demToy1GPmRNA.csv').values
    t_sorted = np.argsort(temp[:, 0], kind='mergesort')
    # toydata = torch.load(Path(data_dir) / 'intermediate_toydata.pt')
    params_list = list()
    orig_data = list()
    num_samples = toydata[0]['samples'].shape[0]
    x_observed = torch.tensor(temp[t_sorted, 0:2]).permute(1, 0)

    for i in range(len(toydata)):
        params = torch.tensor([toydata[i][key] for key in ['l1', 'l2', 'sensitivity', 'decay', 'diffusion']])
        samples = toydata[i]['samples']
        for sample in range(num_samples):
            lf = samples[sample, 1681:]
            out = samples[sample, :1681]
            lf_out = torch.stack([lf, out], dim=0)
            orig_data.append(lf_out)
            params_list.append(params)
    params = torch.stack(params_list)
    orig_data = torch.stack(orig_data)
    shuffle = torch.randperm(orig_data.size()[0])
    orig_data = orig_data[shuffle]
    params = params[shuffle]
    torch.save({'x_observed': x_observed, 'orig_data': orig_data, 'params': params}, Path(data_dir) / 'toydata.pt')


if __name__ == '__main__':
    dataset = HomogeneousReactionDiffusion(data_dir='./data/', nn_format=False)
    tx = torch.tensor(dataset.orig_data[0]).t()
    with torch.no_grad():
        tot = 4 * 4 * 5 * 5
        i = 0
        objects = list()
        for sensitivity in np.linspace(0.2, 0.9, 6):
            for l1 in np.linspace(0.1, 0.4, 5):
                for l2 in np.linspace(0.1, 0.4, 5):
                    for diffusion in np.linspace(0.001, 0.1, 6):
                        for decay in np.linspace(0.01, 0.4, 5):
                            kernel = ReactionDiffusionGenerator(
                                lengthscale=[l1, l2],
                                decay=decay,
                                diffusion=diffusion,
                                sensitivity=sensitivity
                            )
                            Kuu, Kyy, Kyu, Kuy = kernel.joint(tx, tx)
                            kern = torch.zeros((2 * 1681, 2 * 1681))
                            kern[:1681, :1681] = Kyy
                            kern[:1681, 1681:] = Kyu
                            kern[1681:, :1681] = Kuy
                            kern[1681:, 1681:] = Kuu
                            try:
                                eigval, eigvec = torch.symeig(kern, eigenvectors=True)
                            except:
                                print('Failed for', l1, l2, sensitivity, decay, diffusion)
                                continue
                            eps = -1e-5
                            num = torch.sum((~(eigval >= eps)).type(torch.int)).item()
                            if num > 30:
                                print('Failed for', l1, l2, sensitivity, decay, diffusion)
                                continue
                            eigval_root = eigval.clamp_min(0.0).sqrt()
                            corr_matrix = (eigvec * eigval_root).transpose(-1, -2)
                            a = torch.randn(torch.Size([50, 2 * 1681]))
                            samples = a @ corr_matrix
                            obj = {
                                'samples': samples.clone(),
                                'sensitivity': sensitivity,
                                'l1': l1,
                                'l2': l2,
                                'diffusion': diffusion,
                                'decay': decay
                            }
                            objects.append(obj)
                            i += 1
                            print('Done ', i, '/', tot)

        # torch.save(, 'intermediate_toydata.pt')
        save_dataset(objects)
        print('Saved')
