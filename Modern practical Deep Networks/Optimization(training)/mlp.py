import numpy as np

def forward(X, W0, b0, W1, b1):
    # sigmoid
    A1 = 1 / (1 + np.exp(-(X.dot(W0) + b0)))

    # relu
    #A1 = X.dot(W0) + b0
    #A1[A1 < 0] = 0
    #print('A1: {}'.format(A1))

    U = A1.dot(W1) + b1
    expU = np.exp(U)
    Y = expU / expU.sum(axis=1, keepdims=True)
    return A1, Y

def J_derivative_W1(T, Y, A1):
    return A1.T.dot(Y - T)

def J_derivative_b1(T, Y):
    return (Y - T).sum(axis=0)

def J_derivative_W0(T, Y, W1, A1, X):
    return X.T.dot(((Y - T).dot(W1.T) * (A1 * (1 - A1)))) # for sigmoid
    #return X.T.dot(((Y - T).dot(W1.T) * (A1 > 0))) # for relu

def J_derivative_b0(T, Y, W1, A1):
    return ((Y - T).dot(W1.T) * (A1 * (1 - A1))).sum(axis=0) # for sigmoid
    #return ((Y - T).dot(W1.T) * (A1 > 0)).sum(axis=0) # for relu
