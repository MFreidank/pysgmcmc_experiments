#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-

from sacred import Experiment
from sacred.observers import MongoObserver
import numpy as np
from itertools import product

from pysgmcmc.models.bayesian_neural_network import BayesianNeuralNetwork
from pysgmcmc.models.objective_functions import sinc
from pysgmcmc.optimizers.sghmc import SGHMC
from pysgmcmc.optimizers.sghmchd import SGHMCHD

from utils import package_versions, init_random_uniform

experiment = Experiment("Bayesian Neural Network: 'sinc' fit.")
experiment.observers.append(MongoObserver.create(db_name="BNN_sinc"))

OPTIMIZERS = {"SGHMC": SGHMC, "SGHMCHD": SGHMCHD}


@experiment.main
def fit_bnn(sampler, stepsize, _rnd, _seed, num_training_datapoints=20):
    x_train = init_random_uniform(
        np.zeros(1), np.ones(1), num_points=num_training_datapoints,
        rng=np.random.RandomState(seed=_seed)
    )
    y_train = sinc(x_train)
    x_test = np.linspace(0, 1, 100)[:, None]
    y_test = sinc(x_test)

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
    stepsizes = (1e-9, 1e-7, 1e-5, 1e-3, 0.01)
    samplers = tuple(OPTIMIZERS.keys())

    for sampler, stepsize in product(tuple(OPTIMIZERS.keys()), stepsizes):
        experiment.run(config_updates={"sampler": sampler, "stepsize": stepsize})
