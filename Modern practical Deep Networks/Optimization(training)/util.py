import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

## Preprocess raw data ##
# Get PCA reduced train & test data (previously centered around 0)
def get_pca_normalized_data():
    print('Getting and normalizing PCA reduced inputs...')
    df = pd.read_csv('./Datasets/MNIST_data/train.csv')
    data = df.as_matrix().astype(np.float32)
    np.random.shuffle(data)
    
    X = data[:, 1:]
    t = data[:,0]
    X_train = X[:-1000]
    t_train = t[:-1000]
    X_test = X[-1000:]
    t_test = t[-1000:]

    # Center the inputs around the mean of the training inputs
    mu = X_train.mean(axis=0)
    X_train = X_train - mu
    X_test = X_test - mu

    # Transform the centered inputs
    pca = PCA()
    X_pca_train = pca.fit_transform(X_train)
    X_pca_test = pca.transform(X_test)
    
    # Justify taking the top 300 principal components
    # Because it contains over 95% of the variance of the original data
    plot_cumulative_variance(pca)

    # Reduce: take the 300 first PCA features
    X_pca_train = X_pca_train[:, :300]
    X_pca_test = X_pca_test[:, :300]

    # Normalize 
    mu_pca = X_pca_train.mean(axis=0)
    std_pca = X_pca_train.std(axis=0)
    X_pca_train = (X_pca_train - mu_pca) / std_pca
    X_pca_test = (X_pca_test - mu_pca) / std_pca

    return X_pca_train, X_pca_test, t_train, t_test

def get_normalized_data():
    print('Getting and normalizing inputs...') 
    df = pd.read_csv('./Datasets/MNIST_data/train.csv')
    data = df.as_matrix().astype(np.float32)
    np.random.shuffle(data)
    X = data[:,1:]
    t = data[:,0]
    
    X_train = X[:-1000]
    t_train = t[:-1000]
    X_test = X[-1000:]
    t_test = t[-1000:]

    # Normalize the inputs around the mean of the training inputs
    mu_train = X_train.mean(axis=0)
    std_train = X_train.std(axis=0)
    np.place(std_train, std_train == 0, 1)
    X_train = (X_train - mu_train) / std_train
    X_test = (X_test - mu_train ) / std_train

    return X_train, X_test, t_train, t_test

def plot_cumulative_variance(pca):
    P = []
    for p in pca.explained_variance_ratio_:
        if (len(P) == 0):
            P.append(p)
        else:
            P.append(p + P[-1])
    plt.plot(P)
    plt.savefig('cumulative_variance_pca.png')
    return P

## Prediction and learning tools
# Softmax multi-classification
def forward(X, W, b):
    Z = X.dot(W) + b
    expZ = np.exp(Z)
    Y = expZ / expZ.sum(axis=1, keepdims=True)
    return Y

def predict(Y):
    return np.argmax(Y, axis=1)

def accuracy(pred, t):
    return np.mean(pred == t)

def J(T, Y):
    J_n = -T * np.log(Y) 
    return J_n.sum()

def gradW(T, Y, X):
    return X.T.dot(Y - T)

def gradb(T, Y):
    return (Y-T).sum(axis=0)

# K = 10: digits from 0 to 9
def T_indicator(t):
    N = len(t)
    t = t.astype(np.int32)
    T = np.zeros((N, 10))
    for n in range(N):
        T[n, t[n]] = 1
    return T

def benchmark_full():
    X_train, X_test, t_train, t_test = get_normalized_data()
    print('Performing multi-class regression on normalized inputs...')
    N, D = X_train.shape
    T_train = T_indicator(t_train)
    T_test = T_indicator(t_test)
    K = 10 #10 digits
    
    # Initialize weights
    W = np.random.randn(D, K) / np.sqrt(D)
    b = np.zeros(K)
    
    J_train = []
    J_test = []
    CR_test = []
    
    print('Gradient-descent: 500 iterations - lr = 0.00004 - reg = 0.01')
    lr = 0.00004
    reg = 0.01
    for epoch in range(500):
        # predict
        Y_train = forward(X_train, W, b)
        Y_test = forward(X_test, W, b)
        # Save the cost
        j_train = J(T_train, Y_train)
        J_train.append(j_train)
        J_test.append(J(T_test, Y_test))
        # Save the accuracy on test data
        err = accuracy(predict(Y_test), t_test)
        CR_test.append(err)
        # Learn from your mistakes
        W -= lr * (gradW(T_train, Y_train, X_train) - reg * W)
        b -= lr * (gradb(T_train, Y_train) - reg * b)
        # Have a look at cost and accuracy evolutions
        if epoch % 10 == 0:
            print('Cost on train data at epoch {}: {}'.format(epoch, j_train))
            print('Model accuracy on test data: ', err)
    
    Y_test_final = forward(X_test, W, b)
    final_accuracy = accuracy(predict(Y_test_final), t_test)
    print('Final accuracy on test data: {}'.format(final_accuracy))  
    iters = range(len(J_train))
    plt.plot(iters, J_train, iters, J_test)
    plt.savefig('benchmark_full_epochs=500_lr=0.00004_reg=0.01.png')
    plt.plot(CR_test)
    plt.savefig('accuracy_test_data_full.png')

def benchmark_pca():
    X_train, X_test, t_train, t_test = get_pca_normalized_data()
    print('Performing multi-class regression on pca reduced normalized inputs (300 principal components saved)...')
    N, D = X_train.shape
    T_train = T_indicator(t_train)
    T_test = T_indicator(t_test)
    K = 10 #10 digits
    
    # Initialize weights
    W = np.random.randn(D, K) / np.sqrt(D)
    b = np.zeros(K)
    
    J_train = []
    J_test = []
    CR_test = []
    
    print('Gradient-descent: 500 iterations - lr = 0.0001 - reg = 0.01')
    # pca top 300 --> err = 0.07
    lr = 0.0001
    reg = 0.01
    for epoch in range(500):
        # predict
        Y_train = forward(X_train, W, b)
        Y_test = forward(X_test, W, b)
        # Save the cost
        j_train = J(T_train, Y_train)
        J_train.append(j_train)
        J_test.append(J(T_test, Y_test))
        # Save the accuracy on test data
        err = accuracy(predict(Y_test), t_test)
        CR_test.append(err)
        # Learn from your mistakes
        W -= lr * (gradW(T_train, Y_train, X_train) - reg * W)
        b -= lr * (gradb(T_train, Y_train) - reg * b)
        # Have a look at cost and accuracy evolutions
        if epoch % 10 == 0:
            print('Cost on train data at epoch {}: {}'.format(epoch, j_train))
            print('Model accuracy on test data:', err)
    
    Y_test_final = forward(X_test, W, b)
    final_accuracy = accuracy(predict(Y_test_final), t_test)
    print('Final accuracy on test data: {}'.format(final_accuracy))  
    iters = range(len(J_train))
    plt.plot(iters, J_train, iters, J_test)
    plt.savefig('benchmark_pca_epochs=200_lr=0.0001_reg=0.01.png')
    plt.plot(CR_test)
    plt.savefig('accuracy_test_data_pca.png')

#if __name__ == '__main__':
#benchmark_pca()
#benchmark_full()

