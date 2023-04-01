import numpy as np

class MyPCA():
    
    def __init__(self, num_reduced_dims):
        self.dims = num_reduced_dims
        self.w = None

    def fit(self, X):
        # data was normalized already so I'm not doing it here
        # same in autoencoder

        # get covariance matrix 
        # rowvar=False just makes it so rows are data points and columns are features
        # it tried to allocate 26GB when I didn't lol
        X_cov = np.cov(X, rowvar=False)

        # get eigenvalues 
        e_vals, e_vecs = np.linalg.eig(X_cov)

        # discard imaginary part
        e_vecs = np.real(e_vecs)
        e_vals = np.real(e_vals)

        # get eigenvectors corresponding to top self.dim dimensions
        inds = np.argsort(e_vals)[::-1]

        # extract top self.dim eigenvectors
        self.w = e_vecs[:,inds][:,:self.dims]

    def project(self, x):
        # do the actual projection
        return x.dot(self.w)