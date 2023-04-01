################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np

import torch
import torch.nn as nn

import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize

from matplotlib import pyplot as plt

from MyAutoencoder import MyAutoencoder

from hw4_utils import load_MNIST, plot_points

np.random.seed(2023)

batch_size = 10

normalize_vals = (0.1307, 0.3081)

# load MNIST dataset
train_dataset, test_dataset, train_loader, test_loader = load_MNIST(batch_size, normalize_vals)

#####################
# ADD YOUR CODE BELOW
#####################

auto_enc = MyAutoencoder()

losses = auto_enc.fit(train_loader, 50)

images = []
labels = []

# get the 1000 images from the test set and plot

for _ in range(1000):
    image, label = test_dataset[torch.randint(len(test_dataset), (1,)).item()]

    images.append(image)
    labels.append(label)

images = torch.stack(images)

reduced, replicated = auto_enc.transform(images)

reduced_np = reduced.detach().numpy()

plt.plot(losses)
plt.title("Epoch Losses")
plt.show()

fig, ax = plt.subplots()
for g in np.unique(labels):
    i = np.where(labels == g)
    ax.scatter(reduced_np[i,0], reduced_np[i,1], label=g)
ax.legend()

plt.show()