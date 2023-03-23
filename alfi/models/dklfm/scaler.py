class Scaler:
    stds = None
    means = None
    n_feats = 2
    do_scale = True

    def scale(self, x, inverse=False, indices=None):
        """x shape (n_task, n_feats, T)"""
        if not self.do_scale:
            return x
        if inverse:
            stds = self.stds if indices is None else self.stds[indices]
            means = self.means if indices is None else self.means[indices]
            return (x * stds) + means
        else:
            self.means = x.mean(dim=2).unsqueeze(-1)
            self.stds = x.std(dim=2).unsqueeze(-1)
            print(x.shape, self.means.shape)
            return (x - self.means) / self.stds

    def inv_scale(self, x):
        return self.scale(x, inverse=True)
