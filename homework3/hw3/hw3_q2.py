################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np

import torch
import torch.nn as nn

import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize

from MyMLP import MyMLP

from hw3_utils import load_MNIST

np.random.seed(2023)

batch_size = 32

normalize_vals = (0.1307, 0.3081)

# load MNIST dataset
train_dataset, test_dataset, train_loader, test_loader = load_MNIST(batch_size, normalize_vals)

#####################
# ADD YOUR CODE BELOW
#####################
    
criterion  = nn.CrossEntropyLoss() 
lrs = [
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1
]

optims = [
    "sgd",
    "adagrad",
    "rmsprop",
    "adam"
]

for optim in optims:
    for eta in lrs:
        mlp = MyMLP(
            input_size=28*28, 
            hidden_size=128, 
            output_size=10,
            learning_rate=eta, 
            max_epochs=10
        )

        optimizer = None
        match optim:
            case "sgd":
                optimizer = torch.optim.SGD(mlp.parameters(), lr=eta)
            case "adagrad":
                optimizer = torch.optim.Adagrad(mlp.parameters(), lr=eta)
            case "rmsprop":
                optimizer = torch.optim.RMSprop(mlp.parameters(), lr=eta)
            case "adam":
                optimizer = torch.optim.Adam(mlp.parameters(), lr=eta)
            case _:
                print("Shouldn't happen")
                exit()

        train_loss, train_err = mlp.fit(train_loader, criterion, optimizer)
        test_loss, test_err = mlp.predict(test_loader, criterion)

        print(train_loss)
        print(train_err)
        print(test_loss)
        print(test_err)

