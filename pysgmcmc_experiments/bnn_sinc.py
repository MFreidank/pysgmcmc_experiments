#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-

from sacred import Experiment
from sacred.observers import MongoObserver
import numpy as np
from os.path import expanduser
from itertools import product
import logging

import sys
sys.path.insert(0, expanduser("~/pysgmcmc_pytorch/"))
from pysgmcmc.models.bayesian_neural_network import BayesianNeuralNetwork
from pysgmcmc.models.objective_functions import sinc
from pysgmcmc.optimizers.sghmc import SGHMC
from pysgmcmc.optimizers.sghmchd import SGHMCHD

from utils import package_versions, init_random_uniform

experiment = Experiment("Bayesian Neural Network: 'sinc' fit.")
experiment.observers.append(MongoObserver.create(db_name="BNN_sinc"))

OPTIMIZERS = {"SGHMC": SGHMC, "SGHMCHD": SGHMCHD}


@experiment.main
def fit_bnn(sampler, stepsize, _rnd, _seed, data_seed, num_training_datapoints=20):
    x_train = init_random_uniform(
        np.zeros(1), np.ones(1), num_points=num_training_datapoints,
        rng=np.random.RandomState(seed=data_seed)
    )
    y_train = sinc(x_train)

    x_test = np.linspace(0, 1, 100)[:, None]
    y_test = sinc(x_test)

    model = BayesianNeuralNetwork(
        optimizer=OPTIMIZERS[sampler], lr=stepsize,
        logging_configuration={
            "level": logging.WARN, "datefmt": "y/m/d"
        },
    )

    model.train(x_train, y_train)
    prediction_mean, prediction_variance = model.predict(x_test)

    prediction_std = np.sqrt(prediction_variance)

    return {
        "x_train": x_train.tolist(), "y_train": y_train.tolist(),
        "x_test": x_test.tolist(), "y_test": y_test.tolist(),
        "prediction_mean": prediction_mean.tolist(),
        "prediction_std": prediction_std.tolist(),
        "packages": package_versions()
    }

if __name__ == "__main__":
    stepsizes = (1e-9, 1e-7, 1e-5, 1e-3, 1e-2)
    samplers = tuple(OPTIMIZERS.keys())

    data_seed = np.random.randint(0, 100)

    for sampler, stepsize in product(tuple(OPTIMIZERS.keys()), stepsizes):
        experiment.run(config_updates={
            "sampler": sampler, "stepsize": stepsize, "data_seed": data_seed
        })
