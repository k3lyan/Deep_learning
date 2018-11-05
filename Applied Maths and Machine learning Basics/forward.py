import numpy as np
import matplotlib.pyplot as plt

# Number of samples per class 
N_class = 500

X1 = np.random.randn(N_class, 2) + np.array([0, -2])
X2 = np.random.randn(N_class, 2) + np.array([2, 2])
X3 = np.random.randn(N_class, 2) + np.array([2, -2])
# Concatenate, axis = 0 (add lines)
X = np.vstack([X1, X2, X3])

T = np.array([0] * N_class + [1] * N_class + [2] * N_class)

plt.scatter(X[:,0], X[:,1], c=T, s=100, alpha=0.5)
plt.savefig('inputs.png')

#layers sizes
D = 2
M = 3
K = 3

# Weights
W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M, K)
b2 = np.random.randn(K)

def forward(X, W1, b1, W2, b2):
    # First layer
    Z1 = X.dot(W1) + b1
    A1 = 1 / (1+np.exp(-Z1))
    # Second layer
    Z2 = A1.dot(W2) + b2
    expZ = np.exp(Z2)
    Y = expZ / expZ.sum()
    return Y

def classification_rate(T, P):
    n_correct = 0
    n_total = 0
    for i in range(len(T)):
        n_total += 1
        if (T[i] == P[i]):
            n_correct += 1
    return float(n_correct) / n_total

Y = forward(X, W1, b1, W2, b2)
P = np.argmax(Y, axis = 1)

assert(len(T) == len(P))
print(T)
print(P)

print('Classification rate for random weights: {}'.format(classification_rate(T, P)))


