import torch
import numpy as np

from lafomo.datasets import HomogeneousReactionDiffusion, ReactionDiffusionGenerator


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
                                continue
                            eps = -1e-5
                            num = torch.sum((~(eigval >= eps)).type(torch.int)).item()
                            if num > 30:
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

        torch.save(objects, 'toydata.pt')
        print('Saved')
