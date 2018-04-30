from collections import OrderedDict
from itertools import product
from keras.datasets import boston_housing as BostonHousing
import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver

from pysgmcmc.models.bayesian_neural_network import BayesianNeuralNetwork
from pysgmcmc.models.dataset_wrappers import (
    WineQualityRed, YachtHydrodynamics, Concrete
)
from pysgmcmc.optimizers.sghmc import SGHMC
from pysgmcmc.optimizers.sghmchd import SGHMCHD

experiment = Experiment("BNN_UCI")
experiment.observers.append(
    MongoObserver.create(
        db_name=experiment.get_experiment_info()["name"]
    )
)


@experiment.config
def configuration_setup():
    raise NotImplementedError()

DATASETS = OrderedDict((
    ("Boston Housing", BostonHousing),
    ("Yacht Hydrodynamics", YachtHydrodynamics),
    ("Concrete", Concrete),
    ("Wine Quality Red", WineQualityRed)
))

SAMPLERS = {"SGHMC": SGHMC, "SGHMCHD": SGHMCHD}


@experiment.main
def fit_bnn(sampler, stepsize, _rnd, _seed, dataset,
            burn_in_steps=5000, num_steps=15000, num_nets=100,
            batch_size=32, test_split=0.1):
    (x_train, y_train), (x_test, y_test) = DATASETS[dataset].load_data(
        test_split=test_split, seed=_seed
    )

    model = BayesianNeuralNetwork(
        optimizer=SAMPLERS[sampler],
        n_steps=num_steps,
        burn_in_steps=burn_in_steps, num_nets=num_nets,
        batch_size=batch_size
    )

    model.train(x_train, y_train)
    prediction_mean, prediction_variance = model.predict(x_test)
    prediction_std = np.sqrt(prediction_variance)

    return {
        "x_train": x_train, "y_train": y_train,
        "x_test": x_test, "y_test": y_test,
        "prediction_mean": prediction_mean, "prediction_std": prediction_std
    }


if __name__ == "__main__":
    stepsizes = (1e-9, 1e-7, 1e-5, 1e-3, 0.01)
    samplers = tuple(SAMPLERS.keys())

    for sampler, stepsize in product(tuple(SAMPLERS.keys()), stepsizes):
        experiment.run(config_updates={"sampler": sampler, "stepsize": stepsize})
