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
from pysgmcmc.optimizers.sghmchd_new import SGHMCHD

from utils import package_versions

experiment = Experiment("BNN_UCI")
experiment.observers.append(
    MongoObserver.create(
        db_name=experiment.get_experiment_info()["name"]
    )
)


DATASETS = OrderedDict((
    ("Boston Housing", BostonHousing),
    ("Yacht Hydrodynamics", YachtHydrodynamics),
    ("Concrete", Concrete),
    ("Wine Quality Red", WineQualityRed)
))

OPTIMIZERS = {"SGHMC": SGHMC, "SGHMCHD": SGHMCHD}


@experiment.main
def fit_bnn(sampler, stepsize, data_seed, dataset,
            burn_in_steps=5000, num_steps=15000, num_nets=100,
            batch_size=32, test_split=0.1):
    (x_train, y_train), (x_test, y_test) = DATASETS[dataset].load_data(
        test_split=test_split, seed=data_seed
    )

    model = BayesianNeuralNetwork(
        optimizer=OPTIMIZERS[sampler],
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
        "prediction_mean": prediction_mean, "prediction_std": prediction_std,
        "packages": package_versions()
    }


if __name__ == "__main__":
    stepsizes = (1e-9, 1e-7, 1e-5, 1e-3, 1e-2)
    samplers = tuple(OPTIMIZERS.keys())

    configurations = product(
        tuple(DATASETS.keys()), tuple(OPTIMIZERS.keys()), stepsizes
    )

    data_seed = np.random.randint(0, 10000)

    for dataset, sampler, stepsize in configurations:
        experiment.run(
            config_updates={
                "sampler": sampler,
                "stepsize": stepsize,
                "data_seed": data_seed
            }
        )
