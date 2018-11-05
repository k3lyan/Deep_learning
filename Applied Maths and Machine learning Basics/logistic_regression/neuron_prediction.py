import numpy as np

# Input features
D = 2
# Number of samples
N = 100
# Inputs
X = np.random.randn(N,D)

# Weights
# Bias term directly added in X, axis=1 to add at each input sample
ones = np.array([[1]*N]).T
Xb = np.concatenate((ones, X), axis = 1)

# (D+1,1) because you have to add one row for the biases
w = np.random.randn(D + 1)

# NEURON = FEED (sum) -FORWARD (logistic-binary-regression)
z = Xb.dot(w)
def sigmoid(z):
    return 1/(1+np.exp(-z))

print('Neuron output: {}'.format(sigmoid(z)))
print('Binary (logistic regression) classification: {}'.format(np.round(sigmoid(z))))
