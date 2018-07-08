# Plot (and write to table?) autocorrelation and ESS for samplers with different
# stepsizes on some standard functions (no noise?). See also MCMCVisualizer
# plots and pymc3 plots.

# List of samplers:
# - SGHMC (sgmcmc, theano)
# - SGHMCHD (pysgmcmc, keras)
# - SGLD (pysgmcmc, tensorflow) <- from tensorflow_probability
# - NUTS (pymc3, pymc3) <- compare to NUTS here as well!
# - SVGD (?)
# - ?

# XXX: TODO: Add more energy functions, take: https://github.com/chi-feng/mcmc-demo/blob/master/main/MCMC.j://github.com/chi-feng/mcmc-demo/blob/master/main/MCMC.js
# for inspiration!
# XXX: TODO: Commit changes to samplers (sghmchdSampler, sghmcSampler) in
# pysgmcmc and pull in this repo (also on the cluster!)
import sys
from os.path import dirname, join as path_join
sys.path.insert(0, path_join(dirname(__file__), ".."))
sys.path.insert(0, path_join(dirname(__file__), "robo"))
sys.path.insert(0, path_join(dirname(__file__), "pysgmcmc_development"))
import numpy as np
from keras import backend as K
from collections import OrderedDict
from itertools import islice, product

from pysgmcmc.samplers.energy_functions import (
    to_negative_log_likelihood,
    Banana, Gmm1, Gmm2, Gmm3, MoGL2HMC,
    StandardNormal,
    Donut, Squiggle,
)
from pysgmcmc.samplers.sghmchd import SGHMCHDSampler
from pysgmcmc.samplers.sgld import SGLDSampler

from pysgmcmc_experiments.experiment_wrapper import to_experiment


ENERGY_FUNCTIONS = OrderedDict((
    ("banana",
     (Banana(), lambda: [K.random_normal_variable(mean=0., scale=1., shape=(1,)),
                         K.random_normal_variable(mean=0., scale=1., shape=(1,))])),
    ("gmm1",
     (Gmm1(), lambda: [K.variable(K.random_normal((1,)), name="x", dtype=K.floatx())],)),
    ("gmm2", (
        Gmm2(),
        lambda: [K.variable(K.random_normal((1,)), name="x", dtype=K.floatx())],
    )),
    ("gmm3",
     (Gmm3(), lambda: [K.variable(K.random_normal((1,)), name="x", dtype=K.floatx())],)
     ),
    ("mogl2hmc",
     (MoGL2HMC(), lambda: [K.variable(K.random_normal((1,)), name="x", dtype=K.floatx())],)
     ),
    ("standard_normal",
     (StandardNormal(), lambda: [K.variable(K.random_normal((1,)), name="x", dtype=K.floatx())],)
     ),
    ("donut",
     (Donut(), lambda: [K.random_normal_variable(mean=0., scale=1., shape=(1,)),
                         K.random_normal_variable(mean=0., scale=1., shape=(1,))])),
    ("squiggle",
     (Squiggle(), lambda: [K.random_normal_variable(mean=0., scale=1., shape=(1,)),
                         K.random_normal_variable(mean=0., scale=1., shape=(1,))])),
))

SAMPLERS = OrderedDict((
    # ("SGHMC", SGHMCSampler),  # FIXME : use theano sampler from sgmcmc here!
    ("SGHMCHD", SGHMCHDSampler),
    ("SGLD", SGLDSampler),
))

STEPSIZES = tuple((
    1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.25, 0.5, 1.0,
))

CONFIGURATIONS = tuple((
    {"energy_function": energy_function, "sampler": sampler, "stepsize": stepsize}
    for energy_function, sampler, stepsize in
    product(ENERGY_FUNCTIONS, SAMPLERS, STEPSIZES)
))


def get_trace(sampler, stepsize, energy_function, burn_in_steps=3000, sampling_steps=10 ** 4):
    energy_function_, initial_guess = ENERGY_FUNCTIONS[energy_function]
    initial_sample = initial_guess()
    sampler_cls = SAMPLERS[sampler]

    def loss_for(sampler, energy_function):
        # XXX: Case distinction for different samplers?
        def loss_fun(sample):
            loss_tensor = to_negative_log_likelihood(energy_function)(sample)
            for param in sample:
                param.hypergradient = K.gradients(loss_tensor, param)

            return loss_tensor
        return loss_fun

    loss = loss_for(sampler_cls, energy_function_)(initial_sample)
    sampler = sampler_cls(params=initial_sample, loss=loss, lr=lr)

    _ = list(islice(sampler, burn_in_steps))  # noqa
    samples = np.asarray([sample for _, sample in islice(sampler, sampling_steps)])

    num_steps, num_parameters, num_chains = samples.shape
    samples = np.reshape(samples, (num_chains, num_steps, num_parameters))

    return {"samples": samples.tolist()}

experiment = to_experiment(
    experiment_name="energy_functions",
    function=get_trace,
    configurations=CONFIGURATIONS,
)
