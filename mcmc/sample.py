from multiprocessing import Pool
import reggae
import tensorflow as tf

class ChainResult(object):
    def __init__(self, acceptance_rates, samples):
        self.acceptance_rates = acceptance_rates
        self.samples = samples

def create_chains(model, args, sample_kwargs, num_chains=4, current_state=None):
    '''
    This function creates multiple MCMC chains, runs them in parallel, and blocks until
    the execution is complete.
    
    Args:
        model: The model to initialise with `args` (TODO: not currently used).
        args: The arguments given to model.
        sample_kwags: The keyword arguments given to `model.sample()`.
        num_chains: The number of chains to execute.
        current_state: `list` of `ChainResult` objects to initialise each chain with.
            Default value: None
    Return:
        A `list` of `ChainResult`s
    '''
    results = list()
    with Pool(num_chains) as p:
        for i in range(num_chains):
            results.append(p.apply_async(run_job, [args, sample_kwargs]))

        res = [result.get() for result in results]


    return res


def run_job(args, sample_kwargs):
    with tf.device('/device:cpu:0'):
        model = reggae.models.transcription_mh.TranscriptionMCMC(*args)
        model.sample(**sample_kwargs)
        return ChainResult(model.acceptance_rates, model.samples)
