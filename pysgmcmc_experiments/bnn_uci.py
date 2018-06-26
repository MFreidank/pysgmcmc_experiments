import sys
from os.path import dirname, join as path_join
sys.path.insert(0, path_join(dirname(__file__), "robo"))
sys.path.insert(0, path_join(dirname(__file__), "pysgmcmc_development"))
from itertools import product

import numpy as np
from keras.datasets import boston_housing as BostonHousing

from pysgmcmc.optimizers.sghmc2 import SGHMC as Keras_SGHMC
from pysgmcmc.models.bayesian_neural_network import (
    BayesianNeuralNetwork as KerasBayesianNeuralNetwork
)
from pysgmcmc.models.dataset_wrappers import (
    WineQualityRed, YachtHydrodynamics, Concrete
)
from pysgmcmc_experiments.experiment_wrapper import to_experiment
# XXX: Add robo bnn with robo sghmc to samplers as well.


SAMPLERS = {
    "Keras_SGHMC": Keras_SGHMC,
    "Theano_SGHMC": "sghmc",  # XXX for theano/robo bnn
    # "Keras_SGHMCHD": SGHMCHD XXX Add all variants of SGHMCHD optimization we want to try here => these are considered individual samplers
}

num_repetitions = 10
DATA_SEEDS = list(range(num_repetitions))

STEPSIZES = (1e-9, 1e-7, 1e-5, 1e-3, 1e-2, 5e-2, 1e-1)


CONFIGURATIONS = tuple((
    {"sampler": sampler, "stepsize": stepsize, "data_seed": data_seed}
    for data_seed, sampler, stepsize in product(DATA_SEEDS, SAMPLERS, STEPSIZES)
))


def fit_uci(sampler, stepsize, data_seed,
            burn_in_steps=5000, num_steps=15000, num_nets=100,
            batch_size=32, test_split=0.1):

    datasets = (BostonHousing, YachtHydrodynamics, Concrete, WineQualityRed)

    results = {}

    for dataset in datasets:
        (x_train, y_train), (x_test, y_test) = dataset.load_data(
            test_split=test_split, seed=data_seed
        )
        if sampler.startswith("Keras"):
            sampler_cls = SAMPLERS[sampler]
            bnn = KerasBayesianNeuralNetwork(
                optimizer=sampler_cls,
                n_steps=num_steps,
                burn_in_steps=burn_in_steps, num_nets=num_nets,
                batch_size=batch_size,
                **sampler_kwargs
            )
        elif sampler.startswith("Theano"):
            raise NotImplementedError("theano sampler not yet supported")
        else:
            raise NotImplementedError(sampler)
        bnn.train(x_train, y_train)
        prediction_mean, prediction_variance = bnn.predict(x_test)

        results[dataset.__name__] = {
            "prediction_mean": prediction_mean,
            "prediction_variance": prediction_variance
        }

    return results


experiment = to_experiment(
    experiment_name="bnn_uci",
    function=fit_uci,
    configurations=CONFIGURATIONS,
)
