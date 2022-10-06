from gpytorch.mlls import VariationalELBO


class MaskedVariationalELBO(VariationalELBO):

    def _log_likelihood_term(self, variational_dist_f, target, mask=None, weight=None, **kwargs):
        log_prob = self.likelihood.expected_log_prob(target, variational_dist_f, **kwargs)
        if weight is not None:
            log_prob *= weight
        if mask is not None:
            log_prob = log_prob[mask]
        return log_prob.sum(-1)
