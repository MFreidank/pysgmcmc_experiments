#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
# XXX: Turn into sacred.experiment
import torch
from torchvision import datasets, transforms
from os.path import dirname, expanduser, join as path_join

import sys
sys.path.insert(0, expanduser("~/pysgmcmc_pytorch/"))
sys.path.insert(0, path_join(dirname(__file__), ".."))

from pysgmcmc.models.bayesian_neural_network import BayesianNeuralNetwork
from pysgmcmc.optimizers.sghmc import SGHMC
from pysgmcmc.optimizers.sghmchd import SGHMCHD

from utils import network_architectures


architectures = network_architectures(num_classes=10)

net = architectures["alexnet"]

# net = lambda: __import__("torchvision").models.alexnet(num_classes=10)
# net = lambda: models.resnet152()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

ds = datasets.CIFAR10(
    "~/data/CIFAR10",
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
)


train_loader = torch.utils.data.DataLoader(
    ds, batch_size=1, shuffle=True
)

optimizer = SGHMCHD
bnn = BayesianNeuralNetwork(
    optimizer=optimizer,
    lr=1e-2,
    network_architecture=net
)

bnn.train_on_dataset(ds)
