import numpy as np
import pandas as pd
from sklearn import svm

def get_best_acc(err):
    best = err.iloc[err[2].argmax()]
    return best[0], best[1]

def dataset3Params(X, y, Xval, yval):
    """returns your choice of C and sigma. You should complete
    this function to return the optimal C and sigma based on a
    cross-validation set.
    """

# You need to return the following variables correctly.

# ====================== YOUR CODE HERE ======================
# Instructions: Fill in this function to return the optimal C and sigma
#               learning parameters found using the cross validation set.
#               You can use svmPredict to predict the labels on the cross
#               validation set. For example, 
#                   predictions = svmPredict(model, Xval)
#               will return the predictions on the cross validation set.
#
#  Note: You can compute the prediction error using 
#        mean(double(predictions ~= yval))
#
    c_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sig_list = c_list
    err = []
    for c_val in c_list:
        for sig_val in sig_list:
            gamma = 1.0 / (2.0 * sig_val ** 2)
            clf = svm.SVC(C=c_val, kernel='rbf', gamma=gamma, max_iter=200).fit(X, y)
            e = clf.score(Xval, yval)
            err.append([c_val, sig_val, e])
    return get_best_acc(pd.DataFrame(err))

# =========================================================================
