import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from process_data import get_data

def T_indicator(t, K):
    N = len(t)
    ind = np.zeros((N,K))
    for n in range(N):
        ind[n, t[n]] = 1
    return ind

# Get the data
X, t = get_data()
X, t = shuffle(X, t)
t = t.astype(np.int32)
D = X.shape[1]
K = len(set(t))

X_train = X[:-100,:]
t_train = t[:-100]
T_train = T_indicator(t_train, K)

X_test = X[-100:,:]
t_test = t[-100:]
T_test = T_indicator(t_test, K)

W = np.random.randn(D, K)
b = np.zeros(K)

def softmax(z):
    return 1 / (1 + np.exp(-z))

def forward(X, W, b):
    return softmax(X.dot(W) + b)

# P = P_y=k_given_x_joint_to_parameters
def P(Y):
    return np.argmax(Y, axis = 1)

# classification rate averaged on the N samples at the epoch e 
def classification_rate(t, P):
    return np.mean(t == P)

# cross_entropy = -log(L), where L = P(X|Y,parameters) = likelihood
def cross_entropy(T, Y):
    return -np.mean(T*np.log(Y))

train_costs = []
test_costs = []
learninig_rate = 0.001

# Gradient-descent
for epoch in range(100000):
    Y_train = forward(X_train, W, b)
    Y_test = forward(X_test, W, b)
    c_train = cross_entropy(T_train, Y_train)
    c_test = cross_entropy(T_test, Y_test)
    train_costs.append(c_train)
    test_costs.append(c_test)
    
    W -= learninig_rate * X_train.T.dot((Y_train - T_train))
    b -= learninig_rate * (Y_train - T_train).sum(axis=0)
    if (epoch % 1000 == 0):
        print('{}:\t cost_train: {}\t cost_test: {}'.format(epoch, c_train, c_test))

print('Final train classification_rate: {}'.format(classification_rate(t_train, P(Y_train))))
print('Final test classification_rate: {}'.format(classification_rate(t_test, P(Y_test))))

legend1, = plt.plot(train_costs, label='train cost')
legend2, = plt.plot(test_costs, label='test cost')
plt.legend([legend1, legend2])
plt.savefig('logistic_softmax_training.png')
