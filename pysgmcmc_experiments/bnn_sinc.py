import sys
from os.path import dirname, join as path_join
sys.path.insert(0, path_join(dirname(__file__), ".."))
sys.path.insert(0, path_join(dirname(__file__), "robo"))
sys.path.insert(0, path_join(dirname(__file__), "pysgmcmc_development"))
from itertools import product
from pysgmcmc_experiments.experiment_wrapper import to_experiment

import numpy as np

from pysgmcmc.models.bayesian_neural_network import BayesianNeuralNetwork
from pysgmcmc.models.objective_functions import sinc

from pysgmcmc.optimizers.sghmchd4 import SGHMCHD
# from pysgmcmc.optimizers.sghmc2 import SGHMC
from robo.models.bnn import BayesianNeuralNetwork as Robo_BNN

from utils import init_random_uniform

SAMPLERS = {
    "SGHMC": "sghmc",
    "SGHMCHD": SGHMCHD,
}

num_repetitions = 10
DATA_SEEDS = list(range(num_repetitions))

STEPSIZES = (1e-9, 1e-7, 1e-5, 1e-3, 1e-2)
CONFIGURATIONS = tuple((
    {"sampler": sampler, "stepsize": stepsize, "data_seed": data_seed}
    for data_seed, sampler, stepsize in product(DATA_SEEDS, SAMPLERS, STEPSIZES)
))

CONFIGURATIONS = tuple(
    configuration for configuration in CONFIGURATIONS if configuration["sampler"] == "SGHMCHD"
)


def fit_sinc(sampler, stepsize, data_seed, num_training_datapoints=20):
    x_train = init_random_uniform(
        np.zeros(1), np.ones(1), num_points=num_training_datapoints,
        rng=np.random.RandomState(seed=data_seed)
    )
    y_train = sinc(x_train)

    x_test = np.linspace(0, 1, 100)[:, None]
    y_test = sinc(x_test)

    if sampler == "SGHMC":
        model = Robo_BNN(sampling_method=SAMPLERS[sampler], l_rate=stepsize)
    else:
        model = BayesianNeuralNetwork(
            optimizer=SAMPLERS[sampler], learning_rate=stepsize,
        )

    model.train(x_train, y_train)
    prediction_mean, prediction_variance = model.predict(x_test)

    prediction_std = np.sqrt(prediction_variance)

    return {
        "prediction_mean": prediction_mean.tolist(),
        "prediction_std": prediction_std.tolist(),
        "x_train": x_train.tolist(), "y_train": y_train.tolist(),
        "x_test": x_test.tolist(), "y_test": y_test.tolist()
    }

experiment = to_experiment(
    experiment_name="sinc",
    function=fit_sinc,
    configurations=CONFIGURATIONS,
)
