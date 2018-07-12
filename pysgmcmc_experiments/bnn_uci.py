import sys
from os.path import dirname, join as path_join
sys.path.insert(0, path_join(dirname(__file__), ".."))
sys.path.insert(0, path_join(dirname(__file__), "robo"))
sys.path.insert(0, path_join(dirname(__file__), "pysgmcmc_development"))
import numpy as np

from robo.models.bnn import BayesianNeuralNetwork as Robo_BNN

from itertools import product

from keras.datasets import boston_housing as BostonHousing

from pysgmcmc.optimizers.sghmchd4 import SGHMCHD
from pysgmcmc.callbacks import TensorBoard
from pysgmcmc.models.bayesian_neural_network import (
    BayesianNeuralNetwork as KerasBayesianNeuralNetwork
)

from pysgmcmc.models.dataset_wrappers import (
    WineQualityRed, YachtHydrodynamics, Concrete
)
from pysgmcmc_experiments.experiment_wrapper import to_experiment
from keras.losses import kullback_leibler_divergence


SAMPLERS = {
    "sghmc": "sghmc",
    "SGHMCHD": SGHMCHD
}

num_repetitions = 10
DATA_SEEDS = list(range(num_repetitions))

STEPSIZES = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2,)


CONFIGURATIONS = tuple((
    {"sampler": sampler, "stepsize": stepsize, "data_seed": data_seed}
    for data_seed, sampler, stepsize in product(DATA_SEEDS, SAMPLERS, STEPSIZES)
))


def fit_uci(sampler, stepsize, data_seed, burn_in_steps=5000,
            num_steps=15000, num_nets=100, batch_size=32, test_split=0.1):
    datasets = (BostonHousing, YachtHydrodynamics, Concrete, WineQualityRed)

    results = {}

    for dataset in datasets:
        train_data, (x_test, y_test) = dataset.load_data(
            test_split=test_split, seed=data_seed
        )
        had_nans = True

        while had_nans:
            if sampler == "sghmc":
                model = Robo_BNN(
                    l_rate=stepsize,
                    sampling_method="sghmc", n_nets=num_nets, burn_in=burn_in_steps,
                    n_iters=num_steps, bsize=batch_size
                )
            elif sampler.startswith("SGHMCHD"):
                # SGHMCHD approaches with different kwargs

                model = KerasBayesianNeuralNetwork(
                    optimizer=SAMPLERS[sampler], learning_rate=stepsize,
                    train_callbacks=(TensorBoard(histogram_freq=1, batch_size=20, ),),
                    hyperloss=lambda y_true, y_pred: kullback_leibler_divergence(y_true=y_true, y_pred=y_pred[:, 0])
                )
            else:
                raise NotImplementedError()

            model.train(*train_data)
            prediction_mean, prediction_variance = model.predict(x_test)

            had_nans = np.isnan(prediction_mean).any() or np.isnan(prediction_variance).any()

        results[dataset.__name__] = {
            "x_test": x_test.tolist(),
            "y_test": y_test.tolist(),
            "prediction_mean": prediction_mean.tolist(),
            "prediction_variance": prediction_variance.tolist()
        }

    return results


experiment = to_experiment(
    experiment_name="uci",
    function=fit_uci,
    configurations=CONFIGURATIONS,
)
