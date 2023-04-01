################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np

from matplotlib import pyplot as plt

from MyPCA import MyPCA

from hw4_utils import load_MNIST, convert_data_to_numpy, plot_points

np.random.seed(2023)

normalize_vals = (0.1307, 0.3081)

batch_size = 100

# load MNIST dataset
train_dataset, test_dataset, train_loader, test_loader = load_MNIST(batch_size, normalize_vals)

# convert to numpy
X, y = convert_data_to_numpy(train_dataset)

#####################
# ADD YOUR CODE BELOW
#####################

pca = MyPCA(num_reduced_dims=2)

pca.fit(X)
X_proj = pca.project(X)

fig, ax = plt.subplots()
for g in np.unique(y):
    i = np.where(y == g)
    ax.scatter(X_proj[i,0], X_proj[i,1], label=g)
ax.legend()
plt.show()