from sacred import Experiment
from sacred.observers import MongoObserver

from itertools import islice, product

from pysgmcmc.samplers.sghmc import SGHMCSampler
from pysgmcmc.samplers.sghmchd import SGHMCHDSampler
from pysgmcmc.samplers.sgld import SGLDSampler
from pysgmcmc.samplers.energy_functions import (
    Banana, Gmm1, Gmm2, Gmm3, MoGL2HMC, to_negative_log_likelihood
)

experiment = Experiment("Energy_Functions")
experiment.observers.append(
    MongoObserver.create(experiment.get_experiment_info()["name"])
)

ENERGY_FUNCTIONS = {
    "banana": (Banana(), None),
    "gmm1": (Gmm1(), None),
    "gmm2": (Gmm2(), None),
    "gmm3": (Gmm3(), None),
    "mog-l2hmc": (MoGL2HMC(), None),
}

SAMPLERS = {
    "SGHMC": SGHMCSampler, "SGHMCHD": SGHMCHDSampler, "SGLD": SGLDSampler,
}


@experiment.main
def get_chains(sampler, stepsize, energy_function, num_chains, samples_per_chain,
               burn_in_steps, keep_every, _seed):
    function, initial_guess = ENERGY_FUNCTIONS[energy_function]

    def chain():
        initial_sample = initial_guess()
        sampler_object = SAMPLERS[sampler](
            loss=to_negative_log_likelihood(function),
            params=initial_sample,
            lr=stepsize, burn_in_steps=burn_in_steps,
        )
        num_steps = burn_in_steps + samples_per_chain * keep_every
        return [
            sample for _, sample
            in islice(sampler_object, burn_in_steps, num_steps, keep_every)
        ]

    return {
        "chains": [chain() for _ in range(num_chains)]
    }


if __name__ == "__main__":
    stepsizes = (1e-9, 1e-7, 1e-5, 1e-3, 0.01)

    num_chains = 2
    burn_in_steps = 3000
    samples_per_chain = 10 ** 4
    keep_every = 1

    configurations = product(SAMPLERS.keys(), stepsizes, ENERGY_FUNCTIONS.keys())

    for sampler, stepsize, energy_function in configurations:
        experiment.run(
            config_updates={
                "sampler": sampler, "stepsize": stepsize,
                "energy_function": energy_function,
                "num_chains": num_chains, "samples_per_chain": samples_per_chain,
                "burn_in_steps": burn_in_steps, "keep_every": keep_every
            }
        )
