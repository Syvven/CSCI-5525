import numpy as np

class MySVM:

    def __init__(self, d, max_iters, eta, c):
        self.d = d
        self.iters = max_iters
        self.lr = eta
        self.c = c
        self.w = None
        self.b = None

    def fit(self, X, y):
        self.w = np.random.uniform(-0.01, 0.01, X.shape[1])
        self.b = np.random.uniform(-0.01, 0.01)

        for _ in range(self.iters):
            for i in range(len(X)):
                pass
        

    def predict(self, X):
        preds = np.inner(self.w, X) + self.b
        return np.sign(preds)
        