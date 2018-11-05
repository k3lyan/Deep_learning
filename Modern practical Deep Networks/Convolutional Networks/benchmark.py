import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from scipy.io import loadmat #to load matlab SVHN files
from sklearn.utils import shuffle
from datetime import datetime

def accuracy(p, t):
    return np.mean(p == t)

def flatten(X):
    # inputs: 32*32*3*N
    # Here D = H*W*C from input as we're not using
    # any convpool layer, thus D = 3072 
    # outputs: N*3072
    N = X.shape[-1]
    flat = np.zeros((N, 3072)) 
    for i in range(N):
        flat[i] = X[:,:,:,i].reshape(3072)
    return flat

def get_data():
    if not os.path.exists('../Datasets/SVHN_data/train_32x32.mat'):
        print('SVHN Dataset is not foundable.')
        print('Please download it at hhtp://ufldl.stanford.edu/housenumbers.')
        print('Place train_32X32.mat and test_32x32.mat files in ../Datasets/SVHN_data/ (from the current directory).')
        exit()

    train = loadmat('../Datasets/SVHN_data/train_32x32.mat')
    test = loadmat('../Datasets/SVHN_data/test_32x32.mat')
    return train, test

#train, test = get_data()
#print(train)

#Epoch 9 batch_indx 130: validation cost = 22619.943359375 - validation accuracy = 0.74 %

def main():
    train, test = get_data()

    # We need to flatten and scale the inputs
    # Don't let it having values fron 0 to 255
    X_train = flatten(train['X'].astype(np.float32) / 255.)
    # Flatten the output and makes it go from 0 to 9 labels
    # while matlab ranges it from 1 to 10
    t_train = train['y'].flatten() - 1
    X_train, t_train = shuffle(X_train, t_train)
    X_test = flatten(test['X'].astype(np.float32) / 255.)
    t_test = test['y'].flatten() - 1
    
    # Dimensionnality
    N, D = X_train.shape
    M1 = 1000
    M2 = 500
    K = 10
    epochs = 20
    print_period = 10
    batch_size = 500
    nb_batches = N // batch_size

    # Weights initialization
    W0_init = np.random.randn(D, M1) / np.sqrt(D+M1)
    b0_init = np.zeros(M1)
    W1_init = np.random.randn(M1, M2) / np.sqrt(M2+M1)
    b1_init = np.zeros(M2)
    W2_init = np.random.randn(M2, K) / np.sqrt(K+M1)
    b2_init = np.zeros(K) 
    
    # TF environment
    X_pl = tf.placeholder(tf.float32, shape=(None, D), name='X')
    t_pl = tf.placeholder(tf.int32, shape=(None,), name='t')
    W0 = tf.Variable(W0_init.astype(np.float32))
    b0 = tf.Variable(b0_init.astype(np.float32))
    W1 = tf.Variable(W1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))

    A1 = tf.nn.relu(tf.matmul(X_pl, W0) + b0)
    A2 = tf.nn.relu(tf.matmul(A1, W1) + b1)
    Z3 = tf.matmul(A2, W2) + b2
    
    J = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits = Z3,
                labels = t_pl
                )
            )
    
    train_op = tf.train.RMSPropOptimizer(0.0001, decay=0.99, momentum=0.9).minimize(J)
    y = tf.argmax(Z3, 1)

    t0 = datetime.now()
    test_data_costs = []

    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)
        for epoch in range(epochs):
            for batch_indx in range(nb_batches):
                X_batch = X_train[batch_indx*batch_size:(batch_indx+1)*batch_size,]
                t_batch = t_train[batch_indx*batch_size:(batch_indx+1)*batch_size,]
                session.run(train_op, feed_dict={X_pl:X_batch,t_pl:t_batch})
                if batch_indx % print_period == 0:
                    j_test = session.run(J, feed_dict={X_pl:X_test,t_pl:t_test})
                    test_data_costs.append(j_test)
                    y_test = session.run(y, feed_dict={X_pl:X_test})
                    acc_test = accuracy(y_test, t_test)
                    print('Epoch {} batch_indx {}: validation cost = {} - validation accuracy = {}%'.format(epoch, batch_indx, j_test, acc_test))
        j_test_final = session.run(J, feed_dict={X_pl:X_test, t_pl:t_test})
        y_test_final = session.run(y, feed_dict={X_pl:X_test})
        final_accuracy = accuracy(y_test_final, t_test)
        print('Final test cost: {}'.format(j_test_final))
        print('Final test accuracy: {}'.format(final_accuracy*100))
    print('Elapsed time: {}'.format(datetime.now() - t0))
    plt.plot(J)
    plt.show()

if __name__ == '__main__':
    main()

