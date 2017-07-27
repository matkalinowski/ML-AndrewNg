import numpy as np
from numpy.linalg import svd

def pca(X):
    """computes eigenvectors of the covariance matrix of X
      Returns the eigenvectors U, the eigenvalues (on diagonal) in S
    """

    # You need to return the following variables correctly.

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should first compute the covariance matrix. Then, you
    #               should use the "svd" function to compute the eigenvectors
    #               and eigenvalues of the covariance matrix.
    #
    # Note: When computing the covariance matrix, remember to divide by m (the
    #       number of examples).
    #
    m, n = X.shape
    cov_mat = np.dot(X.T, X) / m
    U, S, V = svd(cov_mat, compute_uv=True)
    return U, np.diag(S), V

    # =========================================================================