import numpy as np
import matplotlib.pyplot as plt
from pre_process import get_binary_data
from sklearn.utils import shuffle

def sigmoid(z):
    return 1 /(1 + np.exp(-z))

def feedforward(X, W, b):
    return sigmoid(X.dot(W)+b)

def cross_entropy(T, Y):
    return -np.mean(T*np.log(Y)+ (1 - T)*np.log(1 - Y))

def classification_rate(T, P_round):
    return np.mean(T == P_round)

def training():
    X, T = get_binary_data()
    
    # ex:100
    nb_test = int(input('How many samples for the test validation ?'))
    X_train = X[:-nb_test]
    T_train = T[:-nb_test]

    X_test = X[-nb_test:]
    T_test = T[-nb_test:]

    D = X.shape[1]
    W = np.random.randn(D)
    b = 0
    
    train_costs = []
    test_costs = []
    #ex: 0.001
    learning_rate = float(input('Which learning rate for the gradient-descent training ?'))
    #ex: 10000
    I = int(input('How many iterations for the training ?'))
    for i in range(I):
        Y_train = feedforward(X_train, W, b)
        Y_test = feedforward(X_test, W, b)
        cost_train = cross_entropy(T_train, Y_train)
        cost_test = cross_entropy(T_test, Y_test)
        train_costs.append(cost_train)
        test_costs.append(cost_test)
        if ((i+1) % 1000 == 0):
            print('{} iterations\t cost training samples:{}\t cost test samples:{}'.format(i+1, cost_train, cost_test))
        W -= learning_rate*X_train.T.dot(Y_train - T_train)
        b -= learning_rate*(Y_train - T_train).sum()
    
    print('Final train classification rate: {}'.format(classification_rate(T_train, np.round(Y_train))))
    print('Final test classification rate: {}'.format(classification_rate(T_test, np.round(Y_test))))
    legend1, = plt.plot(train_costs, label = 'TRAIN COSTS')        
    legend2, = plt.plot(test_costs, label = 'TESTS COSTS')
    plt.legend([legend1, legend2])
    plt.savefig('alpha({})_iterations({}).png'.format(learning_rate, I))
   
training()

