################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np

import torch
import torch.nn as nn

import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize

from MyCNN import MyCNN

from hw3_utils import load_MNIST

np.random.seed(2023)

batch_size = 32

normalize_vals = (0.1307, 0.3081)

# load MNIST dataset
train_dataset, test_dataset, train_loader, test_loader = load_MNIST(batch_size, normalize_vals)

#####################
# ADD YOUR CODE BELOW
#####################

import matplotlib.pyplot as plt

cnn = MyCNN(
    input_channels=1,
    output_size=10,
    epochs=100
)

criterion  = nn.CrossEntropyLoss() 
optim = torch.optim.SGD(cnn.parameters(), lr=0.1)

train_loss, train_err = cnn.fit(train_loader, criterion, optim)
incorrect_samples, test_loss, test_err = cnn.predict(test_loader, criterion)

print(train_loss)
print(train_err)
print(test_loss)
print(test_err)

fig = plt.figure(figsize=(10, 7))
rows = 1
columns = 5
  
# Adds a subplot at the 2nd position
for i,(pic, pred, label) in enumerate(incorrect_samples):
    fig.add_subplot(rows, columns, i+1)
  
    # showing image
    plt.imshow(pic.permute(1, 2, 0))
    plt.title(f"Actual: {label}, Pred: {pred}")

plt.show()