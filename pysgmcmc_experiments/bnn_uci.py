import logging
from os.path import dirname, join as path_join
from collections import OrderedDict
from itertools import product
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver

from pysgmcmc.models.bayesian_neural_network import BayesianNeuralNetwork
from pysgmcmc.data.datasets import (
    BostonHousing, WineQualityRed, YachtHydrodynamics, Concrete
)
from pysgmcmc.optimizers.sghmc import SGHMC
from pysgmcmc.optimizers.sghmchd import SGHMCHD

from utils import package_versions

experiment = Experiment("BNN_UCI")
experiment.observers.append(
    FileStorageObserver.create(
        path_join(dirname(__file__), "..", "results", "bnn_uci")
    )
)


DATASETS = OrderedDict((
    ("Boston Housing", BostonHousing),
    ("Yacht Hydrodynamics", YachtHydrodynamics),
    ("Concrete", Concrete),
    ("Wine Quality Red", WineQualityRed)
))

SAMPLERS = {"SGHMC": SGHMC, "SGHMCHD": SGHMCHD}


@experiment.main
def fit_bnn(sampler, stepsize, _rnd, _seed, dataset,
            burn_in_steps=5000, num_steps=15000,
            batch_size=32, test_split=0.1):
    (x_train, y_train), (x_test, y_test) = DATASETS[dataset].load_data(
        test_split=test_split, seed=_seed
    )

    model = BayesianNeuralNetwork(
        optimizer=SAMPLERS[sampler],
        num_steps=num_steps,
        burn_in_steps=burn_in_steps,
        batch_size=batch_size,
        logging_configuration={"level": logging.WARN}
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
    stepsizes = (1e-9, 1e-7, 1e-5, 1e-3, 1e-2, 5e-2, 8e-2, 1e-1)
    samplers = tuple(SAMPLERS.keys())

    configurations = product(
        tuple(DATASETS.keys()), tuple(SAMPLERS.keys()), stepsizes
    )

    for dataset, sampler, stepsize in configurations:
        experiment.run(config_updates={
            "sampler": sampler, "stepsize": stepsize, "dataset": dataset
        })
