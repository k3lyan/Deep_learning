import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from util import get_normalized_data, accuracy, J, T_indicator, predict
from mlp import forward, J_derivative_W1, J_derivative_W0, J_derivative_b1, J_derivative_b0

def main():
    max_iter = 20
    print_period = 10

    X_train, X_test, t_train, t_test = get_normalized_data()
    T_train = T_indicator(t_train)
    T_test = T_indicator(t_test)
    
    lr = 0.00004
    reg = 0.01
    N, D = X_train.shape
    batch_sz = 500
    nb_batches = N // batch_sz
    M = 300
    K = 10
    print('N_train = {}\t N_test = 1000\t D = {}\t M = {}\t K = {}\t batch_size = {}\t nb_batches = {}\t lr_cst = {}\n'.format(N, D, M, K, batch_sz, nb_batches, lr))
    # np.sqrt(D) ~ 28
    W0 = np.random.randn(D, M) / 28
    b0 = np.zeros(M)
    W1 = np.random.randn(M, K) / np.sqrt(M)
    b1 = np.zeros(K)
    
    # 1. CONSTANT LEARNING RATE
    print('CONSTANT LEARNING RATE')
    #t0 = datetime.now()
    J_constant_lr = [] # measured on test data every 10 batches
    accuracy_constant_lr =[] # measured on test data every 10 batches
    for epoch in range(max_iter):
        for batch_index in range(nb_batches):
            X_batch = X_train[batch_index*batch_sz:(batch_index+1)*batch_sz,]
            T_batch = T_train[batch_index*batch_sz:(batch_index+1)*batch_sz,]
            
            A_batch, Y_batch = forward(X_batch, W0, b0, W1, b1)

            # Updates
            W1 -= lr * J_derivative_W1(T_batch, Y_batch, A_batch)
            b1 -= lr * J_derivative_b1(T_batch, Y_batch)
            W0 -= lr * J_derivative_W0(T_batch, Y_batch, W1, A_batch, X_batch)
            b0 -= lr * J_derivative_b0(T_batch, Y_batch, W1, A_batch)

            if (batch_index % print_period) == 0:
                _, Y_test = forward(X_test, W0, b0, W1, b1)
                j_test = J(T_test, Y_test)
                J_constant_lr.append(j_test)
                acc = accuracy(predict(Y_test), t_test)
                accuracy_constant_lr.append(acc)
                print('Epoch n° {} batch n° {}:\t TEST COST {}\t TEST ACCURACY RATE: {}'.format(epoch, batch_index, j_test, acc))
    _, Y_test_final = forward(X_test, W0, b0, W1, b1)
    print('Final ACCURACY RATE on TEST data: {}\n'.format(accuracy(predict(Y_test_final), t_test)))
    #print('Constant lr execution time: {}\n'.format(datetime.now() - t0))

    # 2. RMSProp
    print('RMSProp')
    #t0 = datetime.now()
    
    W0 = np.random.randn(D, M) / 28
    b0 = np.zeros(M)
    W1 = np.random.randn(M, K) / np.sqrt(M)
    b1 = np.zeros(K)
    
    J_RMSProp = []
    accuracy_RMSProp = []
    
    lr0 = 0.001 #if you set the initial lr too high you'll get Nan
    cache_W1 = 0
    cache_b1 = 0
    cache_W0 = 0
    cache_b0 = 0
    decay = 0.999
    eps = 0.000001
    for epoch in range(max_iter):
        for b_index in range(nb_batches):
            X_batch = X_train[b_index*batch_sz:(b_index+1)*batch_sz,]
            T_batch= T_train[b_index*batch_sz:(b_index+1)*batch_sz,]
            A_batch, Y_batch = forward(X_batch, W0, b0, W1, b1)
            
            # Updates
            gW1 = J_derivative_W1(T_batch, Y_batch, A_batch) + reg * W1 
            cache_W1 = decay * cache_W1 + (1 - decay) * gW1 * gW1
            W1 -= lr / (np.sqrt(cache_W1 + eps)) * gW1 

            gb1 = J_derivative_b1(T_batch, Y_batch) + reg * b1
            cache_b1 = decay * cache_b1 + (1 - decay) * gb1 * gb1
            b1 -= lr/(np.sqrt(cache_b1) + eps) * gb1

            gW0 = J_derivative_W0(T_batch, Y_batch, W1, A_batch, X_batch) + reg * W0
            cache_W0 = decay * cache_b0 + (1 - decay) * gW0 * gW0
            W0 -= lr / (np.sqrt(cache_W0) + eps) * gW0
            
            gb0 = J_derivative_b0(T_batch, Y_batch, W1, A_batch)
            cache_b0 = decay * cache_b0 + (1 - decay) * gb0 * gb0
            b0 -= lr / (np.sqrt(cache_b0) + eps) * gb0

            if (b_index % 10) == 0:
                _, Y_test = forward(X_test, W0, b0, W1, b1)
                j_test = J(T_test, Y_test)
                J_RMSProp.append(j_test)
                acc = accuracy(predict(Y_test), t_test)
                accuracy_RMSProp.append(acc)
                print('Epoch n° {} Batch n°{}:\t TEST COST: {}\t TEST ACCURACY RATE: {}'.format(epoch, b_index*nb_batches, j_test, acc))
    
    _, Y_test_final = forward(X_test, W0, b0, W1, b1)
    print('Final accuracy rate on test data: {}'.format(accuracy(predict(Y_test_final), t_test)))
    #print('Constant lr execution time: {}'.format(datetime.now() - t0))

    plt.plot(J_constant_lr, label ='constant lr')
    plt.plot(J_RMSProp, label='RMSProp')
    plt.legend()
    plt.savefig('RMSProp.py')

if __name__ == '__main__':
    main()





    






