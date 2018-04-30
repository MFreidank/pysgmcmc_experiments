#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
import numpy as np
from itertools import product
from sacred import Experiment
from sacred.observers import MongoObserver

from pysgmcmc.optimizers.sghmc import SGHMC
from pysgmcmc.optimizers.sghmchd import SGHMCHD
from pysgmcmc.optimizers.sgld import SGLD
from pysgmcmc.models.bayesian_neural_network import BayesianNeuralNetwork
from pysgmcmc.models.objective_functions import (
    branin, bohachevski, camelback, goldstein_price,
    rosenbrock, sin_one, sin_two, levy
)

from utils import package_versions, init_random_uniform


experiment = Experiment("BNN_hpolib")
experiment.observers.append(
    MongoObserver.create(
        db_name=experiment.get_experiment_info()["name"]
    )
)

OPTIMIZERS = {"SGHMC": SGHMC, "SGHMCHD": SGHMCHD, "SGLD": SGLD}


OBJECTIVE_FUNCTIONS = {
    "branin": (branin, 2, 100),
    "bohachevski": (bohachevski, 2, 100),
    "camelback": (camelback, 2, 100),
    "goldstein_price": (goldstein_price, 2, 100),
    "rosenbrock": (rosenbrock, 2, 100),
    "sin_two": (sin_two, 2, 100),
    "sin_one": (sin_one, 1, 100),
    "levy": (levy, 2, 100),
}


@experiment.main
def fit_bnn(sampler, stepsize, _rnd, _seed, objective_function,
            objective_function_dimensions, num_training_datapoints=100):
    function = OBJECTIVE_FUNCTIONS[objective_function]

    lower = np.ones(objective_function_dimensions) * -10
    upper = np.pnes(objective_function_dimensions) * 10

    x_train = init_random_uniform(
        lower=lower, upper=upper, num_points=num_training_datapoints,
        rng=np.random.RandomState(_seed)
    )
    y_train = function(x_train)

    if objective_function_dimensions == 1:
        x_test = np.linspace(-10, 10, 1000)[:, None]
    elif objective_function_dimensions == 2:
        x_test = np.asarray(list(product(range(-10, 10), range(-10, 10))))
    else:
        raise NotImplementedError(
            "Objective functions with number of dimensions >= 3 not supported!"
        )
    y_test = np.asarray([function(x) for x in x_test])

    # XXX: Grab learning rate at each step as well?
    # This would allow plotting learning curves.
    # XXX: Consider this in other experiments as well!

    model = BayesianNeuralNetwork(
        optimizer=OPTIMIZERS[sampler], learning_rate=stepsize, seed=_seed
    )

    model.train(x_train, y_train)
    prediction_mean, prediction_variance = model.predict(x_test)

    prediction_std = np.sqrt(prediction_variance)

    return {
        "x_train": x_train, "y_train": y_train,
        "x_test": x_test, "y_test": y_test,
        "prediction_mean": prediction_mean, "prediction_std": prediction_std,
        "packages": package_versions()

    }


if __name__ == "__main__":
    stepsizes = (1e-09, 1e-07, 1e-05, 1e-03, 0.01)

    configurations = product(
        tuple(OPTIMIZERS.keys()), stepsizes, tuple(OBJECTIVE_FUNCTIONS.keys())
    )

    for sampler, stepsize, objective_function in configurations:
        _, objective_function_dimensions, num_training_datapoints = OBJECTIVE_FUNCTIONS[objective_function]

        experiment.run(
            config_updates={
                "sampler": sampler, "stepsize": stepsize,
                "objective_function": objective_function,
                "objective_function_dimensions": objective_function_dimensions,
                "num_training_datapoints": num_training_datapoints
            }
        )
