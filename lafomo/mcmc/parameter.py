class Parameter:
    def __init__(self,
                 name,
                 prior,
                 initial_value,
                 transform=None,
                 fixed=False):
        self.name = name
        self.prior = prior
        self.transform = (lambda x: x) if transform is None else transform
        self.value = initial_value
        self.fixed = fixed

    def propose(self, *args):
        if self.fixed:
            return self.value
        assert self.proposal_dist is not None, 'proposal_dist must not be None if you use propose()'
        return self.proposal_dist(*args).sample().numpy()
