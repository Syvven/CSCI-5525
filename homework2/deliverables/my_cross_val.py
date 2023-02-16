import numpy as np

def mse(y_actual, y_pred):
    return np.sum((y_actual - y_pred)**2) / y_actual.shape[0]

def zero_one(y_actual, y_pred):
    return np.sum(y_actual != y_pred) / y_actual.shape[0]

def my_cross_val(model, loss_func, X, y, k=10, ridge=False):
    block_size = X.shape[0] // k;
    # lambda functions so there's not too many if statements in the loop
    if (loss_func == 'mse'):
        err = lambda ya, yhat : mse(ya, yhat)
    if (loss_func == 'err_rate'):
        err = lambda ya, yhat : zero_one(ya, yhat)

    # Created a fold vector to make indexing 
    #  the test and train by k-fold easier
    # There was probably an easier way but whatever
    folds = np.zeros(X.shape[0])
    counter = 0
    for i in range(X.shape[0]):
        folds[i] = counter
        if i % block_size == 0 and i != 0:
            counter += 1

    total_error = 0
    error_array = []
    for iter in range(k):
        # Split into train and test sets based on fold
        train_x = X[folds != iter,:]
        train_y = y[folds != iter]

        test_x = X[folds == iter,:]
        test_y = y[folds == iter]

        # Fit and predict on the passed through model    
        model.fit(train_x, train_y)

        preds = model.predict(test_x)

        # use the lambda function for the error
        error_array.append(err(test_y, preds))
        total_error += error_array[-1] 

    if (ridge): return total_error, error_array
    else: return total_error
        