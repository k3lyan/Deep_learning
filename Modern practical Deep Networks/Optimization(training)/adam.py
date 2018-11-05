# RMSProp with momentum VS Adam

import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from datetime import datetime
from util import get_normalized_data, accuracy, J, T_indicator, predict
from mlp import forward, J_derivative_W1, J_derivative_b1, J_derivative_W0, J_derivative_b0

def main():
    max_iter = 10
    print_period = 10
    X_train, X_test, t_train, t_test = get_normalized_data()
    T_train = T_indicator(t_train)
    T_test = T_indicator(t_test)
    
    # Dimensionality 
    N, D = X_train.shape
    M = 300
    K = 10
    batch_size = 500
    nb_batches = N // batch_size
    print('N:{}\t batch_size: {}\t nb_batches: {}\t  D:{}\t M:{}\t K:{}'.format(N, batch_size, nb_batches, D, M, K))
    
    # hyperparameters
    reg = 0.01
    lr0= 0.001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    print('Hyperparameters: reg:{}\t lr0:{}\t beta1:{}\t beta2:{}\t eps:{}\n'.format(reg, lr0, beta1, beta2, eps))
    
    # Weights initialization
    W0_0 = np.random.randn(D, M) / np.sqrt(D)
    b0_0 = np.zeros(M)
    W1_0 = np.random.randn(M, K) / np.sqrt(M)
    b1_0 = np.zeros(K)
    W0 = W0_0.copy()
    b0 = b0_0.copy()
    W1 = W1_0.copy()
    b1 = b1_0.copy()
    # 1st Moment
    mW0 = 0
    mb0 = 0
    mW1 = 0
    mb1 = 0
    # 2nd Moment
    vW0 = 0
    vb0 = 0
    vW1 = 0
    vb1 = 0

    # 1. Adam
    t0 = datetime.now()
    J_adam = []
    accuracy_adam = []
    t = 1
    for epoch in range(max_iter):
        for batch_index in range(nb_batches):
            X_batch = X_train[batch_index*batch_size: (batch_index+1)*batch_size,]
            T_batch = T_train[batch_index*batch_size: (batch_index+1)*batch_size,]
            A_batch, Y_batch = forward(X_batch, W0, b0, W1, b1)
            # gradient updates
            gW1 = J_derivative_W1(T_batch, Y_batch, A_batch) + reg * W1
            gb1 = J_derivative_b1(T_batch, Y_batch) + reg * b1
            gW0 = J_derivative_W0(T_batch, Y_batch, W1, A_batch, X_batch) + reg * W0
            gb0 = J_derivative_b0(T_batch, Y_batch, W1, A_batch) + reg * b0
            # 1st moment updates
            mW1 = beta1 * mW1 + (1-beta1) * gW1
            mb1 = beta1 * mb1 + (1-beta1) * gb1
            mW0 = beta1 * mW0 + (1-beta1) * gW0
            mb0 = beta1 * mb0 + (1-beta1) * gb0
            # 2nd moment updates
            vW1 = beta2 * vW1 + (1-beta2) * gW1 * gW1
            vb1 = beta2 * vb1 + (1-beta2) * gb1 * gb1
            vW0 = beta2 * vW0 + (1-beta2) * gW0 * gW0
            vb0 = beta2 * vb0 + (1-beta2) * gb0 * gb0
            # corrections
            corr1 = 1 - beta1 ** t
            corr2 = 1 - beta2 ** t
            mW1_c = mW1 / corr1
            mb1_c = mb1 / corr1
            mW0_c = mW0 / corr1
            mb0_c = mb0 / corr1
            vW1_c = vW1 / corr2
            vb1_c = vb1 / corr2
            vW0_c = vW0 / corr2
            vb0_c = vb0 / corr2        
            # t update
            t += 1
            # gradient descent
            W1 -= lr0 * mW1_c / np.sqrt(vW1_c + eps)
            b1 -= lr0 * mb1_c / np.sqrt(vb1_c + eps)
            W0 -= lr0 * mW0_c / np.sqrt(vW0_c + eps)
            b0 -= lr0 * mb0_c / np.sqrt(vb0_c + eps)
            '''
            if (batch_index % print_period) == 0:
                _, Y_validation = forward(X_test, W0, b0, W1, b1)
                j = J(T_test, Y_validation)
                J_adam.append(j)
                acc = accuracy(predict(Y_validation), t_test)
                accuracy_adam.append(acc)
                print('Epoch {}\t batch_index {}\t iteration {} : cost {}\t accuracy: {}\t'.format(epoch, batch_index, epoch * batch_index, j, acc))
            '''
    _, Y_final_test = forward(X_test, W0, b0, W1, b1)
    print('Final accuracy with Adam: {}'.format(accuracy(predict(Y_final_test), t_test)))
    print('Execution time with Adam: {}\n'.format(datetime.now() - t0))

    # 2. RMSProp with momentum
    W0 = W0_0
    b0 = b0_0
    W1 = W1_0
    b1 = b1_0
    
    decay_rate = 0.999
    mu = 0.9

    cW1 = 1
    cb1 = 1
    cW0 = 1
    cb0 = 0
    
    vW1 = 0
    vb1 = 0
    vW0 = 0
    vb1 = 0

    t0 = datetime.now()
    J_rmsprop_momentum = []
    accuracy_rmsprop_momentum = []
    for epoch in range(max_iter):
        for batch_index in range(nb_batches):
            X_batch = X_train[batch_index*batch_size:(batch_index+1)*batch_size,]
            T_batch = T_train[batch_index*batch_size:(batch_index+1)*batch_size,]
            A_batch, Y_batch = forward(X_batch, W0, b0, W1, b1)

            # gradient_update
            gW1 = J_derivative_W1(T_batch, Y_batch, A_batch) + reg * W1
            gb1 = J_derivative_b1(T_batch, Y_batch) + reg * b1
            gW0 = J_derivative_W0(T_batch, Y_batch, W1, A_batch, X_batch) + reg * W0
            gb0 = J_derivative_b0(T_batch, Y_batch, W1, A_batch) + reg * b0
            # cache update
            cW1 = decay_rate * cW1 + (1 - decay_rate) * gW1 * gW1
            cb1 = decay_rate * cb1 + (1 - decay_rate) * gb1 * gb1
            cW0 = decay_rate * cW0 + (1 - decay_rate) * gW0 * gW0
            cb0 = decay_rate * cb0 + (1 - decay_rate) * gb0 * gb0
            # momentum updates
            vW1 = mu * vW1 + (1 - mu) * lr0 * gW1 / np.exp(cW1 + eps)
            vb1 = mu * vb1 + (1 - mu) * lr0 * gb1 / np.exp(cb1 + eps)
            vW0 = mu * vW0 + (1 - mu) * lr0 * gW0 / np.exp(cW0 + eps)
            vb0 = mu * vb0 + (1 - mu) * lr0 * gb0 / np.exp(cb0 + eps)
            # gradient descent
            W1 -= vW1
            b1 -= vb1
            W0 -= vW0
            b0 -= vb0
            '''
            if (batch_index % print_period) == 0:
                _, Y_validation = forward(X_test, W0, b0, W1, b1)
                j = J(T_test, Y_validation)
                J_rmsprop_momentum.append(j)
                acc = accuracy(predict(Y_validation), t_test)
                accuracy_rmsprop_momentum.append(acc)
                print('Epoch {}\t batch_index {}\t iteration {} : cost {}\t accuracy: {}\t'.format(epoch, batch_index, epoch * batch_index, j, acc))
            '''
    _, Y_final_test = forward(X_test, W0, b0, W1, b1)
    print('Final accuracy with RMSProp with momentum: {}'.format(accuracy(predict(Y_final_test), t_test)))
    print('Execution time with RMSProp with momentum: {}\n'.format(datetime.now() - t0))

    plt.plot(J_adam, label='adam')
    plt.plot(J_rmsprop_momentum, label='rmsprop with momentum')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
