# Plot (and write to table?) autocorrelation and ESS for samplers with different
# stepsizes on some standard functions (no noise?). See also MCMCVisualizer
# plots and pymc3 plots.


# Plot the following:
# vs NUTS with different initial stepsizes
# vs HMC with different initial stepsizes
# vs SGHMC with different initial stepsizes
# vs SGLD with different initial stepsizes

# other plots definitely should go in appendix unless they have some super-unexpected results
# => show table with all results (what is a reasonable table format?)

# List of samplers:
# - SGHMC (sgmcmc, theano) [x]
# - SGHMCHD (pysgmcmc, keras) [x]
# - SGLD (pysgmcmc, tensorflow) [x] <- from tensorflow_probability
# - HMC(pymc3) [x]
# - NUTS (pymc3, pymc3) <- compare to NUTS here as well! [x]
# - slice/metropolis (pymc3) [x]

# Left to add:
# - SVGD (?)
# - RelSGHMC (re-implement as optimizer in pysgmcmc)
# -

import sys
from os.path import dirname, join as path_join
sys.path.insert(0, path_join(dirname(__file__), ".."))
sys.path.insert(0, path_join(dirname(__file__), "robo"))
sys.path.insert(0, path_join(dirname(__file__), "pysgmcmc_development"))
import numpy as np
from keras import backend as K
from collections import OrderedDict
from itertools import islice, product

from pysgmcmc.diagnostics import PYSGMCMCTrace
from pysgmcmc.samplers.energy_functions import (
    to_negative_log_likelihood,
    Banana, Gmm1, Gmm2, Gmm3, MoGL2HMC,
    StandardNormal,
    Donut, Squiggle,
)
from pysgmcmc.samplers.sghmc import SGHMCSampler
from pysgmcmc.samplers.sghmchd import SGHMCHDSampler
from pysgmcmc.samplers.sgld import SGLDSampler

from pysgmcmc_experiments.experiment_wrapper import to_experiment
import pymc3 as pm

import dill

# XXX: Handle variable naming properly.
EXPERIMENT_NAME = "energy_functions"


def init_hmc(model, stepsize, init="jitter+adapt_diag", chains=1):
    from pymc3.step_methods.hmc import quadpotential
    if init == 'jitter+adapt_diag':
        start = []
        for _ in range(chains):
            mean = {var: val.copy() for var, val in model.test_point.items()}
            for val in mean.values():
                val[...] += 2 * np.random.rand(*val.shape) - 1
            start.append(mean)
        mean = np.mean([model.dict_to_array(vals) for vals in start], axis=0)
        var = np.ones_like(mean)
        potential = quadpotential.QuadPotentialDiagAdapt(
            model.ndim, mean, var, 10)

        return pm.step_methods.HamiltonianMC(
            step_scale=stepsize,
            potential=potential,
            path_length=1
        )
    else:
        raise NotImplementedError()


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


PYMC3_SAMPLERS = ("NUTS", "HMC", "Metropolis", "Slice",)
SAMPLERS = OrderedDict((
    ("SGHMC", SGHMCSampler),
    ("SGHMCHD", SGHMCHDSampler),
    ("SGLD", SGLDSampler),
    ("NUTS", pm.step_methods.NUTS),
    ("HMC", pm.step_methods.HamiltonianMC),
    ("Metropolis", pm.step_methods.Metropolis),
    ("Slice", pm.step_methods.Slice),
))

STEPSIZES = tuple((
    1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.25, 0.5, 1.0,
))

CONFIGURATIONS = [
    {"energy_function": energy_function, "sampler": sampler, "stepsize": stepsize}
    for energy_function, sampler, stepsize in
    product(ENERGY_FUNCTIONS, SAMPLERS, STEPSIZES)
    if sampler not in ("Metropolis", "Slice")
]

CONFIGURATIONS.extend([
    {"energy_function": energy_function, "sampler": sampler, "stepsize": None}
    for energy_function, sampler in
    product(ENERGY_FUNCTIONS, ("Metropolis", "Slice"))
])


def get_trace(sampler, stepsize, energy_function, _run, burn_in_steps=3000, sampling_steps=10 ** 4, num_chains=10):
    energy_function_, initial_guess = ENERGY_FUNCTIONS[energy_function]
    initial_sample = initial_guess()
    sampler_cls = SAMPLERS[sampler]

    if sampler in PYMC3_SAMPLERS:
        def draw_trace(chain_id):
            with pm.Model() as model:
                energy_function_.to_pymc3()
                if sampler == "NUTS":
                    from pymc3.sampling import init_nuts
                    start, step = init_nuts(
                        init="auto",
                        n_init=200000,
                        model=model,
                        progressbar=True
                    )

                    trace = pm.sample(
                        sampling_steps,
                        tune=burn_in_steps,
                        step=step,
                        chains=1,
                        chain_idx=chain_id,
                        start=start,
                        discard_tuned_samples=False
                    )
                elif sampler == "HMC":
                    step = init_hmc(stepsize=stepsize, model=model)
                    trace = pm.sample(sampling_steps, tune=burn_in_steps, step=step, chains=1, discard_tuned_samples=False)
                else:
                    step = SAMPLERS[sampler]()
                    trace = pm.sample(
                        sampling_steps,
                        tune=burn_in_steps,
                        step=step,
                        chains=1,
                        chain_idx=chain_id,
                        discard_tuned_samples=False
                    )

                return trace

        def combine_traces(multitraces):
            base_trace = multitraces[0]
            for multitrace in multitraces[1:]:
                for chain, strace in multitrace._straces.items():
                    if chain in base_trace._straces:
                        raise ValueError("Chains are not unique.")
                    base_trace._straces[chain] = strace
            return base_trace

        multitrace = combine_traces(
            [draw_trace(chain_id=chain_id) for chain_id in range(num_chains)]
        )

    else:
        def loss_for(sampler, energy_function):
            def loss_fun(sample):
                loss_tensor = to_negative_log_likelihood(energy_function)(sample)
                for param in sample:
                    param.hypergradient = K.gradients(loss_tensor, param)

                return loss_tensor
            return loss_fun

        def draw_chain():
            loss = loss_for(sampler_cls, energy_function_)(initial_sample)
            sampler_ = sampler_cls(params=initial_sample, loss=loss, lr=stepsize)
            samples = np.asarray([
                sample for _, sample in islice(sampler_, burn_in_steps + sampling_steps)
            ])

            if np.isnan(samples).any():
                print("Had nans.. iterating")
                return draw_chain()

            return np.squeeze(samples, 2)

        multitrace = pm.backends.base.MultiTrace([PYSGMCMCTrace(draw_chain(), chain_id=id) for id in range(num_chains)])

    output_filename = path_join(
        dirname(__file__),
        "../results/{}/{}/trace.pkl".format(EXPERIMENT_NAME, _run._id)
    )
    with open(output_filename, "wb") as trace_buffer:
        dill.dump(multitrace, trace_buffer)

    return True

experiment = to_experiment(
    experiment_name=EXPERIMENT_NAME,
    function=get_trace,
    configurations=CONFIGURATIONS,
)
