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
        # set the initial w
        self.w = self.w = np.random.uniform(-0.01, 0.01, X.shape[1])
        self.b = np.random.uniform(-0.01, 0.01)

        # I couldn't really think of a way to speed it up 
        #  like the logistic regression one so its a tad slow
        current_loss = -1
        previous_loss = -2
        grad_w = np.zeros(self.w.shape)
        grad_b = 0
        for _ in range(self.iters-1):
            # check for convergence
            # I put this first so as to more easily do 
            #  this check, rather than having to do stuff before
            if (np.abs(previous_loss - current_loss) < self.d):
                return

            # reset loss stuff
            previous_loss = current_loss
            current_loss = (self.w**2).sum()*0.5

            # update gradients
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

            grad_w = np.zeros(self.w.shape)
            grad_b = 0
            for i in range(len(X)):
                # hinge portion of the obj function
                val = self.c*(1 + -1*y[i]*(np.dot(self.w, X[i]) + self.b))
                hinge = val > 0
                # First gradient for when hinge is non-zero
                if hinge:
                    grad_w += (self.w - self.c * np.dot(y[i], X[i]))
                    grad_b += -1*y[i]
                    current_loss += val*self.c
                # second gradient for when hinge is zero
                else:
                    grad_w += self.w
        

    def predict(self, X):
        # set of linear affine functions w^T*x + b
        preds = np.dot(X, self.w) + self.b
        # only care if its -1 or 1 because those are the labels
        return np.sign(preds)
        