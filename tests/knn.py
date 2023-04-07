import numpy as np

import torch
import torch.nn as nn

import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import Subset

# This is code from a 5525 homework to download and split MNIST

def load_MNIST(batch_size, normalize_vals):

    # for correctly download the dataset using torchvision, do not change!
    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    norm1_val, norm2_val = normalize_vals

 #   transforms = Compose([ToTensor()])
    transforms = Compose([ToTensor()])


    train_dataset = torchvision.datasets.MNIST(root='MNIST-data',
                                               train=True,
                                               download=True,
                                               transform=transforms)

    test_dataset = torchvision.datasets.MNIST(root='MNIST-data', 
                                              train=False,
                                              transform=transforms)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)


    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    return train_dataset, test_dataset, train_loader, test_loader

def convert_data_to_numpy(dataset):
    X = []
    y = []
    for i in range(len(dataset)):
        X.append(dataset[i][0][0].flatten().numpy())# flatten it to 1d vector
        y.append(dataset[i][1])

    X = np.array(X)
    y = np.array(y)

    return X, y

normalize_vals = (0.1307, 0.3081)
train_d, test_d, train_l, test_l = load_MNIST(32, normalize_vals)

np_train_x, np_train_y = convert_data_to_numpy(train_d)
np_test_x, np_test_y = convert_data_to_numpy(test_d)
 
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

train_y = np.zeros_like(np_train_y)
train_y[(np_train_y == 3) | (np_train_y == 7) | (np_train_y == 8)] = 1
test_y = np.zeros_like(np_test_y)
test_y[(np_test_y == 3) | (np_test_y == 7) | (np_test_y == 8)] = 1

knn.fit(np_train_x, train_y)

preds = knn.predict(np_test_x)

print((preds == test_y).sum() / test_y.shape[0])
