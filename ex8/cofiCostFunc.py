import numpy as np

def calc_err_factor(X, Theta, Y, R):
    return (np.dot(X, Theta.T) - Y)* R

def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, Lambda):
    """returns the cost and gradient for the
    """

    # Unfold the U and W matrices from params
    X = np.array(params[:num_movies*num_features]).reshape(num_features, num_movies).T.copy()
    Theta = np.array(params[num_movies*num_features:]).reshape(num_features, num_users).T.copy()


    # You need to return the following values correctly
    J = 0
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost function and gradient for collaborative
    #               filtering. Concretely, you should first implement the cost
    #               function (without regularization) and make sure it is
    #               matches our costs. After that, you should implement the
    #               gradient and use the checkCostFunction routine to check
    #               that the gradient is correct. Finally, you should implement
    #               regularization.
    #
    # Notes: X - num_movies  x num_features matrix of movie features
    #        Theta - num_users  x num_features matrix of user features
    #        Y - num_movies x num_users matrix of user ratings of movies
    #        R - num_movies x num_users matrix, where R(i, j) = 1 if the
    #            i-th movie was rated by the j-th user
    #
    # You should set the following variables correctly:
    #
    #        X_grad - num_movies x num_features matrix, containing the
    #                 partial derivatives w.r.t. to each element of X
    #        Theta_grad - num_users x num_features matrix, containing the
    #                     partial derivatives w.r.t. to each element of Theta


    # =============================================================
    cst = np.power(calc_err_factor(X, Theta, Y, R),2)
    J = np.sum(cst) / 2
    # =============================================================
    lam = Lambda / 2
    X_reg = np.sum(np.power(Theta, 2)) * lam
    Theta_reg = np.sum(np.power(X, 2)) * lam

    J = J + X_reg + Theta_reg

    X_grad = np.dot(calc_err_factor(X, Theta, Y, R), Theta) + (Lambda * X)
    Theta_grad = np.dot(calc_err_factor(X, Theta, Y, R).T, X) + (Lambda * Theta)

    grad = np.hstack((X_grad.T.flatten(),Theta_grad.T.flatten()))

    return J, grad
