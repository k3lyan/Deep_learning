import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from process_data import get_data
import sys

# sys.argv[1] = learning_rate
# sys.argv[2] = iterations

def T_indicator(t, K):
    N = len(t)
    ind = np.zeros((N, K))
    for n in range(N):
        ind[n, t[n]] = 1
    return ind

# Get the data
X, t = get_data()
X, t = shuffle(X, t)
t = t.astype(np.int32)

#N = len(t)
D = X.shape[1]
M = 5
K = len(set(t))

X_train = X[:-100,:]
t_train = t[:-100]
T_train = T_indicator(t_train, K)

X_test = X[-100:,:]
t_test = t[-100:]
T_test = T_indicator(t_test, K)

W1 = np.random.randn(M, K)
b1 = np.zeros(K)
W0 = np.random.randn(D,M)
b0 = np.zeros(M)

def forward(X, W0, b0, W1, b1):
    A1 = np.tanh(X.dot(W0) + b0)
    expU = np.exp(A1.dot(W1) + b1)
    Y = expU / expU.sum(axis=1, keepdims=True)
    return A1, Y
        
def P(Y):
    return np.argmax(Y, axis=1)

def classification_rate(t, P):
    return np.mean(t == P, axis=0)

def cost(T, Y):
    return -np.mean(T * np.log(Y))

def W1_derivative(T, Y, A1):
    return A1.T.dot(Y - T)
        
def b1_derivative(T, Y):
    return (Y - T).sum(axis=0)

def W0_derivative(T, Y, W1, A1, X):
    dA1 = (Y - T).dot(W1.T) * A1 * (1 - A1)
    return X.T.dot(dA1)
    
def b0_derivative(T, Y, W1, A1):
    return ((Y - T).dot(W1.T)*A1*(1-A1)).sum(axis=0)

train_costs = []
test_costs = []
learning_rate = float(sys.argv[1])

# Training
for epoch in range(int(sys.argv[2])):
    A1_train, Y_train = forward(X_train, W0, b0, W1, b1)
    A1_test, Y_test = forward(X_test, W0, b0, W1, b1)
    if((epoch % 1000) == 0):
        c_train = cost(T_train, Y_train)
        c_test = cost(T_test, Y_test)
        train_costs.append(c_train)
        test_costs.append(c_test)
        print('{}:\tcost_train: {}\tcost_test: {}'.format(epoch, c_train, c_test))
    # Gradient-descent
    W1 -= learning_rate * W1_derivative(T_train, Y_train, A1_train)
    b1 -= learning_rate * b1_derivative(T_train, Y_train)
    W0 -= learning_rate * W0_derivative(T_train, Y_train, W1, A1_train, X_train)
    b0 -= learning_rate * b0_derivative(T_train, Y_train, W1, A1_train)

print('Final TRAIN classification rate ({} epochs): {}'.format(sys.argv[2], classification_rate(t_train, P(Y_train))))
print('Final TEST classification rate ({} epochs): {}'.format(sys.argv[2], classification_rate(t_test, P(Y_test))))
legend1, = plt.plot(train_costs, label='train cost')
legend2, = plt.plot(test_costs, label='test cost')
legends = plt.legend([legend1, legend2])
plt.savefig('ANN_alpha({})_epochs({}).png'.format(learning_rate, sys.argv[2]))

