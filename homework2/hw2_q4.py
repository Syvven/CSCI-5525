################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np
import pandas as pd

from MySVM import MySVM

# load dataset
data = pd.read_csv('hw2_q2_q4_dataset.csv', header=None).to_numpy()
X = data[:,:-1]
y = data[:,-1]

# change labels from 0 and 1 to -1 and 1 for SVM
y[y == 0] = -1

num_data, num_features = X.shape

# shuffle dataset
np.random.seed(2023)
perm = np.random.permutation(num_data)

X = X.tolist()
y = y.tolist()

X = [X[i] for i in perm]
y = [y[i] for i in perm]

X = np.array(X)
y = np.array(y)

# append column of 1s to include intercept
X = np.hstack((X, np.ones((num_data, 1))))
num_data, num_features = X.shape

# Split dataset into train and test sets
NUM_TRAIN = int(np.ceil(num_data*0.8))
NUM_TEST = num_data - NUM_TRAIN

X_train = X[:NUM_TRAIN]
X_test = X[NUM_TRAIN:]
y_train = y[:NUM_TRAIN]
y_test = y[NUM_TRAIN:]

#####################
# ADD YOUR CODE BELOW
#####################

# Import your CV package here (either your my_cross_val or sci-kit learn )
from my_cross_val import my_cross_val
from my_cross_val import zero_one

eta_vals = [0.00001, 0.0001, 0.001]
C_vals = [0.01, 0.1, 1, 10, 100]

# SVM
errs = []
for eta_val in eta_vals:
    for c_val in C_vals:

        # instantiate svm object
        svm = MySVM(1e-10, 100, eta_val, c_val)

        # call to CV function to compute error rates for each fold
        total_err = my_cross_val(svm, 'err_rate', X_train, y_train, k=10)
        errs.append(total_err)

        # print error rates from CV
        print(f"Total Error (eta = {eta_val}): {total_err}")

# instantiate svm object for best value of eta and C

# fit model using all training data

# predict on test data

# compute error rate on test data

# print error rate on test data
