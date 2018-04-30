from subprocess import check_output
import re

import numpy as np


def package_versions():
    package_list = check_output(("pip", "list")).decode().split("\n")[2:]

    packages = []

    package_pattern = re.compile(
        "([a-zA-Z0-9-]+) +([^ ]+).*"
    )

    for package in package_list:
        match = package_pattern.search(package)
        if match:
            package, version = match.group(1), match.group(2)
            packages.append((package.strip(), version.strip()))
    return packages


def init_random_uniform(lower, upper, num_points, rng=None):
    if rng is None:
        rng = np.random.RandomState(np.random.randint(0, 10000))

    n_dims = lower.shape[0]

    return np.array(
        [rng.uniform(lower, upper, n_dims) for _ in range(num_points)]
    )
