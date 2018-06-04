from subprocess import check_output
import re

import numpy as np


def network_architectures(num_classes):
    torchvision_models = __import__("torchvision").models
    return {
        "alexnet": lambda: torchvision_models.alexnet(num_classes=num_classes),
        "vgg11": lambda: torchvision_models.vgg11(),
        "vgg11_bn": lambda: torchvision_models.vgg11_bn(),
        "vgg13": lambda: torchvision_models.vgg13(),
        "vgg13_bn": lambda: torchvision_models.vgg13_bn(),
        "vgg16": lambda: torchvision_models.vgg16(),
        "vgg16_bn": lambda: torchvision_models.vgg16_bn(),
        "vgg19": lambda: torchvision_models.vgg19(),
        "vgg19_bn": lambda: torchvision_models.vgg19_bn(),
        "resnet18": lambda: torchvision_models.resnet18(),
        "resnet34": lambda: torchvision_models.resnet34(),
        "resnet50": lambda: torchvision_models.resnet50(),
        "resnet101": lambda: torchvision_models.resnet101(),
        "resnet152": lambda: torchvision_models.resnet152(),
        "squeezenet1_0": lambda: torchvision_models.squeezenet1_0(),
        "squeezenet1_1": lambda: torchvision_models.squeezenet1_1(),
        "densenet121": lambda: torchvision_models.densenet121(),
        "densenet169": lambda: torchvision_models.densenet169(),
        "densenet161": lambda: torchvision_models.densenet161(),
        "densenet201": lambda: torchvision_models.densenet201(),
        "inception_v3": lambda: torchvision_models.inception_v3(),
    }


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

    return np.array(
        [rng.uniform(lower, upper, n_dims) for _ in range(num_points)]
    )


def to_lists(iterable):
    if hasattr(iterable, "__iter__"):
        return list(to_lists(el) for el in iterable)
    else:
        return iterable
