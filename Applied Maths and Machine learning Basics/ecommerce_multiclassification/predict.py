import numpy as np
from process_data import get_data

# Data
X, T = get_data()

# Weights
M = 5
D = X.shape[1]
K = len(set(T))
W1 = np.random.randn(D, M)
b1 = np.zeros(M)
W2 = np.random.randn(M, K)
b2 = np.zeros(K)

def softmax(z):
    expZ= np.exp(z)
    return(expZ/ expZ.sum(axis=1, keepdims=True))

def forward(X, W1, W2, b1, b2):
    Z = X.dot(W1) + b1
    A = np.tanh(Z)
    return(softmax(A.dot(W2 + b2)))

def classification_rate(P, T):
    return np.mean(P == T)

Y = forward(X, W1, W2, b1, b2)
P = np.argmax(Y, axis = 1)

print('Classification rate with random weights: {}'.format(classification_rate(P, T)))
