import torch
from torch.nn.functional import softplus


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]
        # x = x[..., 0]
        y = y.view(num_examples, -1)
        x_mean = x[..., 0].view(num_examples, -1)
        x_var = softplus(x[..., 1]).view(num_examples, -1) + 1e-6
        # print(x.shape, x_mean.shape, x_var.shape)
        diff_norms = torch.norm(x_mean.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y, self.p, 1)
        diff_norms = 0.5 * (torch.log(x_var) + (y - x_mean)**2 / x_var)

        norms = diff_norms.sum(-1)/ y_norms# #

        if self.reduction:
            if self.size_average:
                return torch.mean(norms)
            else:
                return torch.sum(norms)

        return norms

    def __call__(self, x, y):
        return self.rel(x, y)

"""
        y = y.view(num_examples, -1)
        x_mean = x[..., 0].view(num_examples, -1)
        x_var = softplus(x[..., 1]).view(num_examples, -1) + 1e-6

        diff_norms = torch.norm(x_mean.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y, self.p, 1)
        # diff_norms = 0.5 * (torch.log(x_var) + (y - x_mean)**2 / x_var)

        norms = diff_norms/ y_norms#.sum(-1) #

        if self.reduction:
            if self.size_average:
                return torch.mean(norms)
            else:
                return torch.sum(norms)

        return norms
"""