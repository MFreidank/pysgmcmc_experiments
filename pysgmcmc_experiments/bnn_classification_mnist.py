#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import datasets, transforms
from os.path import expanduser
from tqdm import tqdm

import sys
sys.path.insert(0, expanduser("~/pysgmcmc_pytorch/"))
from pysgmcmc.models.bayesian_neural_network import BayesianNeuralNetwork
from pysgmcmc.optimizers.sghmc import SGHMC
from pysgmcmc.optimizers.sghmchd import SGHMCHD


# net = alexnet(num_classes=10)
# net = models.alexnet(num_classes=10)

# net = lambda: alexnet(num_classes=10)
net = lambda: models.resnet18()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

ds = datasets.CIFAR10(
    "~/data/CIFAR10",
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.RandomSizedCrop(224),
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
