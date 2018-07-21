import re
from subprocess import check_output
from os.path import expanduser, join as path_join

import numpy as np



def package_versions():
    package_list = check_output(("pip", "list")).decode().split("\n")[2:]

    packages = []

    package_pattern = re.compile("([a-zA-Z0-9-]+) +([^ ]+).*")

    for package in package_list:
        match = package_pattern.search(package)
        if match:
            package, version = match.group(1), match.group(2)
            packages.append([package.strip(), version.strip()])
    return packages


def init_random_uniform(lower, upper, num_points, rng=None):
    if rng is None:
        rng = np.random.RandomState(np.random.randint(0, 10000))

    n_dims = lower.shape[0]

    return np.array([rng.uniform(lower, upper, n_dims) for _ in range(num_points)])


def format_tex(float_):
    exponent = int(np.floor(np.log10(float_)))
    return "10^{{{}}}".format(exponent)


def rename(name):
    titlecase = {
        "gmm1", "gmm2", "gmm3", "banana", "donut", "squiggle",  # Energy functions
        "bohachevsky", "branin", "hartmann3", "hartmann6", "camelback", "levy", "rosenbrock",  # Hpolib synthetic functions

    }

    if name in titlecase:
        return name.title()

    mappings = {
        "mogl2hmc": "MoGL2HMC", "standard_normal": "Standard Normal",  # Energy functions

        "goldstein_price": "Goldstein Price",
        "sin_one": "sinOne", "sin_two": "sinTwo",  # Hpolib synthetic functions

        "gp": "GP", "gp_mcmc": "GP_MCMC", "RF": "RF", "bohamiann": "BOHAMIANN",
        "keras_bohamiann": "BOHAMIANN_TF", "random_search": "Random Search",
        "entropy_search": "Entropy Search",  # RoBo fmin choices

        "BostonHousing": "Boston Housing",
        "YachtHydrodynamics": "Yacht Hydrodynamics",
        "WineQualityRed": "Wine Quality Red",  # UCI Datasets

        "rmse": "Root mean squared error~(RMSE)",  # UCI benchmark metrics
    }

    return mappings.get(name, name)


def format_performance(mean, stddev, bold=True):
    if bold:
        return "$\mathbf{{{mean}}} \\pm {stddev}$".format(mean=round(mean, 3), stddev=round(stddev, 3))
    return "${mean} \\pm {stddev}$".format(mean=round(mean, 3), stddev=round(stddev, 3))


def format_sampler(sampler, stepsize):
    if stepsize is None:
        return sampler

    return "{sampler} $\epsilon_0 = {stepsize}$".format(
        sampler=sampler.upper(), stepsize=format_tex(stepsize)
    )
