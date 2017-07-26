import numpy as np
import random

def get_random_index_of_ndarray(array):
    return random.randint(0, array.shape[0]-1), \
           random.randint(0, array.shape[1] - 1)

def kMeansInitCentroids(X, K):
    """returns K initial centroids to be
    used with the K-Means on the dataset X
    """

# You should return this values correctly
    centroids = np.zeros((K, X.shape[1]))

# ====================== YOUR CODE HERE ======================
# Instructions: You should set centroids to randomly chosen examples from
#               the dataset X
#
    for i in range(0, K):
        centroids[i] = X[get_random_index_of_ndarray(X)]

# =============================================================
    return centroids
