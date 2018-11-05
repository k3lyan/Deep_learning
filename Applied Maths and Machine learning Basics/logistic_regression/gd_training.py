import numpy as np

N = 100
D = 2

def sigmoid(z):
    return 1 / (1+np.exp(-z))

def cross_entropy(Y, T):
    J = 0
    for i in range(N):
        if(T[i] == 1):
            J -= np.log(Y[i])
        else:
            J -= np.log(1 - Y[i])
    return J

# Inputs
X = np.random.randn(N,D)

# Center the 50 first samples around (X1,X2) = (-2,-2)
X[:50, :] = X[:50, :] - 2*np.ones((50,2))
# Center the 50 first samples around (X1,X2) = (2,2)
X[50:, :] = X[50:, :] + 2*np.ones((50,2))
ones = np.array([N*[1]]).T
Xb = np.concatenate((ones, X), axis=1)

# Targets
T = np.array([0]*50+[1]*50)

# Random initialization
W = np.random.randn(D+1)

# Feed
Z = Xb.dot(W)
# Forward
Y = sigmoid(Z)

W_init = W

print('---- Cross-entropy value going through 100 gradient-descent iterations after random initialization of the weights ----\n')

print('Cross-entropy value without any iteration of gradient-descent: {}'.format(cross_entropy(Y, T)))
learning_rate = 0.1
for i in range(100):
    if ((i+1) % 10 == 0):
        print('Cross-entropy value after {} iterations of gradient-descent: {}'.format(i+1, cross_entropy(Y, T)))
    # Gradient-descent
    W += learning_rate * Xb.T.dot((T - Y))
    # Feedforward
    Y = sigmoid(Xb.dot(W))

W_gd = W
print('\n')

# Closed Bayes form:
W = np.array([0, 4, 4])
Z = Xb.dot(W)
Y = sigmoid(Z)
print('Cross-entropy value (Closed-form Bayes solution for the weigths): {}\n'.format(cross_entropy(Y, T)))

print('W initial: {}'.format(W_init))
print('W Bayes form: {}'.format(W))
print('W after 100 gradient-descent iterations: {}'.format(W_gd))
