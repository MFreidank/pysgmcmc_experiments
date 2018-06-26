import sys
from os.path import dirname, join as path_join
sys.path.insert(0, path_join(dirname(__file__), ".."))
sys.path.insert(0, path_join(dirname(__file__), "robo"))
sys.path.insert(0, path_join(dirname(__file__), "pysgmcmc_development"))
from collections import OrderedDict
from functools import partial
from itertools import product

from pysgmcmc_experiments.experiment_wrapper import to_experiment

import numpy as np
from robo.fmin import (bayesian_optimization, entropy_search, random_search, bohamiann)
from robo.fmin.keras_bohamiann import bohamiann as keras_bohamiann
import hpolib.benchmarks.synthetic_functions as hpobench

BENCHMARKS = OrderedDict((
    ("branin", hpobench.Branin()),
    ("hartmann3", hpobench.Hartmann3()),
    ("hartmann6", hpobench.Hartmann6()),
    ("camelback", hpobench.Camelback()),
    ("goldstein_price", hpobench.GoldsteinPrice()),
    ("rosenbrock", hpobench.Rosenbrock()),
    ("sin_one", hpobench.SinOne()),
    ("sin_two", hpobench.SinTwo()),
    ("bohachevsky", hpobench.Bohachevsky()),
    ("levy", hpobench.Levy())
))

METHODS = OrderedDict((
    ("rf", partial(bayesian_optimization, model_type="rf")),
    ("gp", partial(bayesian_optimization, model_type="gp")),
    ("gp_mcmc", partial(bayesian_optimization, model_type="gp_mcmc")),
    ("entropy_search", entropy_search),
    ("random_search", random_search),
    ("bohamiann", bohamiann),
    ("keras_bohamiann", keras_bohamiann),
))

CONFIGURATIONS = tuple((
    {"benchmark": benchmark, "method": method}
    for benchmark, method in product(BENCHMARKS, METHODS)
))


def optimize_function(benchmark,
                      method,
                      acquisition="log_ei",
                      maximizer="random",
                      num_iterations=200,
                      num_init=2):
    assert benchmark in BENCHMARKS
    assert method in METHODS
    print("{method} on {benchmark}".format(method=method, benchmark=benchmark))

    benchmark_function = BENCHMARKS[benchmark]
    info = benchmark_function.get_meta_information()
    bounds = np.array(info["bounds"])

    method_function = METHODS[method]

    if method == "entropy_search":
        results = method_function(
            benchmark_function, bounds[:, 0], bounds[:, 1],
            num_iterations=num_iterations, n_init=num_init
        )
    elif method == "random_search":
        results = method_function(
            benchmark_function, bounds[:, 0], bounds[:, 1],
            num_iterations=num_iterations,
        )

    else:
        results = method_function(
            benchmark_function, bounds[:, 0], bounds[:, 1],
            num_iterations=num_iterations,
            acquisition_func=acquisition,
            maximizer=maximizer,
            n_init=num_init
        )

    # Offline Evaluation
    regret = []
    for inc in results["incumbents"]:
        r = benchmark_function.objective_function(inc)
        regret.append(r["function_value"] - info["f_opt"])

    print("Logarithm of Final Regret:", np.log(regret))

    return {
        "regret": regret,
    }


experiment = to_experiment(
    experiment_name="hpolib",
    function=optimize_function,
    configurations=CONFIGURATIONS,
)
