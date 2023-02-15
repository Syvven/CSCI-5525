import numpy as np

class MyLogisticRegression:

    def __init__(self, d, max_iters, eta):
        self.d = d
        self.iters = max_iters
        self.lr = eta
        self.w = None

    def fit(self, X, y):
        self.w = np.random.uniform(-0.01, 0.01, X.shape[1])

        preds = np.inner(self.w, X)
        probs = self.sigmoid(preds)

        for it in range(self.iters):
            # use loss function described in the homework page
            # assuming we have to take the average of all of the ones
            # added together because the homework page loss function
            # is only for yi and xi not just y and X

            # i was having issues with divide by zero so had to add some wiggle here
            loss = y*np.log(probs + 1e-15) + (1 - y)*np.log(1 - probs + 1e-15)
            loss = np.mean(loss)

            # Derivation from question 1
            # This is really slow though, 
            #  so a faster version is just below it
            # The faster version is equivalent

            # for j in range(self.w.shape[0]):
            #     total = 0
            #     for i in range(X.shape[0]):
            #         total += self.temp_grad(X[i], y[i], j)
            #     self.w[j] -= self.lr * total

            for j in range(self.w.shape[0]):
                self.w[j] -= self.lr * self.grad(X.T[j], y, X)

            preds = np.inner(self.w, X)
            probs = self.sigmoid(preds)

            # i was having issues with divide by zero so had to add some wiggle here
            new_loss = y*np.log(probs + 1e-15) + (1 - y)*np.log(1 - probs + 1e-15)
            new_loss = np.mean(new_loss)

            # if no change, we have converged, break early
            if (np.abs(new_loss-loss) < self.d):
                break

    # derivation from question 1, the slow version
    def temp_grad(self, xi, yi, j):
        return (
            -yi*xi[j] + (
                (xi[j]*np.exp(np.inner(self.w, xi)))
                /
                (1 + np.exp(np.inner(self.w, xi)))
            )
        )

    # derivation from question 1, the fast version
    def grad(self, xi, yi, X):
        f = np.exp(np.inner(self.w, X))
        return (xi*(-yi + (f / (1 + f)))).sum()

    def predict(self, X):
        # get prediction, use sigmoid to get probabilities
        ret = np.zeros(X.shape[0])
        preds = np.inner(self.w, X)
        probs = self.sigmoid(preds)
        ret[probs >= 0.5] = 1

        return ret

    # normal sigmoid
    def sigmoid(self, vals):
        return 1 / (1 + np.exp(-1*vals))