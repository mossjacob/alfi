import torch
import numpy as np

from lafomo.datasets import ToyReactionDiffusion, ToySpatialTranscriptomics


if __name__ == '__main__':
    dataset = ToySpatialTranscriptomics(data_dir='./data/')
    tx = torch.tensor(dataset.orig_data)
    with torch.no_grad():
        tot = 4 * 4 * 5 * 5
        i = 0
        objects = list()
        for l1 in np.linspace(0.1, 0.4, 4):
            for l2 in np.linspace(0.1, 0.4, 4):
                for diffusion in np.linspace(0.001, 0.1, 5):
                    for decay in np.linspace(0.01, 0.4, 5):
                        kernel = ToyReactionDiffusion(
                            lengthscale=[l1, l2],
                            decay=decay,
                            diffusion=diffusion
                        )
                        Kuu, Kyy, Kyu, Kuy = kernel.joint(tx, tx)
                        kern = torch.zeros((2 * 1681, 2 * 1681))
                        kern[:1681, :1681] = Kyy
                        kern[:1681, 1681:] = Kyu
                        kern[1681:, :1681] = Kuy
                        kern[1681:, 1681:] = Kuu
                        eigval, eigvec = torch.symeig(kern, eigenvectors=True)
                        eigval_root = eigval.clamp_min(0.0).sqrt()
                        corr_matrix = (eigvec * eigval_root).transpose(-1, -2)
                        a = torch.randn(torch.Size([10, 2 * 1681]))
                        samples = a @ corr_matrix

                        obj = {
                            'samples': samples.clone(),
                            'l1': l1,
                            'l2': l2,
                            'diffusion': diffusion,
                            'decay': decay
                        }
                        objects.append(obj)
                        i += 1
                        print('Done ', i, '/', tot)

        torch.save(objects, 'toydata.pt')
        print('Saved')