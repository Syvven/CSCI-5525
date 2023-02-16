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
import math

eta_vals = [0.00001, 0.0001, 0.001]
C_vals = [0.01, 0.1, 1, 10, 100]

# SVM
errs = []
inds = []
i = 0
j = 0

for eta_val in eta_vals:
    for c_val in C_vals:

        # instantiate svm object
        svm = MySVM(1e-10, 100, eta_val, c_val)

        # call to CV function to compute error rates for each fold
        total_err, err_array = my_cross_val(svm, 'err_rate', X_train, y_train, k=10, ridge=True)
        errs.append(total_err)
        inds.append([eta_val, c_val])
        # print error rates from CV
        print(f"Total Error: {total_err}")
        print(f"Eta Val: {eta_val}")
        print(f"C Val: {c_val}")
        print("Error Values For SVM: ")
        print(err_array)
        mean = total_err / len(err_array)
        print(f"Mean error rate From SVM: {mean}")

        stddev = 0
        for i in range(len(err_array)):
            stddev += (err_array[i] - mean)**2

        stddev /= len(err_array)
        stddev = math.sqrt(stddev)

        print(f"Std dev from SVM: {stddev}")

        j+=1
    i+=1

# instantiate svm object for best value of eta and C
best_eta = inds[np.argmin(errs)][0]
best_c = inds[np.argmin(errs)][1]

svm = MySVM(1e-10, 100, best_eta, best_c)

# fit model using all training data
svm.fit(X_train, y_train)

# predict on test data
preds = svm.predict(X_test)

# compute error rate on test data
error = zero_one(y_test, preds)

# print error rate on test data
print(f"Best eta: {best_eta}, Best C: {best_c}\nError: {error}")
