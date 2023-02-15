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
        self.w = self.w = np.random.uniform(-0.01, 0.01, X.shape[1])
        self.b = np.random.uniform(-0.01, 0.01)

        for _ in range(self.iters):
            # for j in range(len(w)):

            for i in range(len(X)):
                condition = (1 + -1*y[i]*(np.dot(self.w, X[i]) + self.b)) <= 0
                if condition:
                    self.w -= self.lr * self.w
                else:
                    self.w -= self.lr * (self.w - self.c * np.dot(y[i], X[i]))
                    self.b -= self.lr * -1*y[i]
        

    def predict(self, X):
        preds = np.inner(self.w, X) + self.b
        return np.sign(preds)
        