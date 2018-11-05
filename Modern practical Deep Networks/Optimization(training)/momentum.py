import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from util import get_normalized_data, accuracy, J, T_indicator, predict
from mlp import forward, J_derivative_W1, J_derivative_W0, J_derivative_b1, J_derivative_b0
from datetime import datetime

def main():
    # compare 3 scenarios:
    # 1. batch SGD
    # 2. batch SGD with momentum
    # 3. batch SGD with Nesterov momentum
    
    # Inputs and targets
    X_train, X_test, t_train, t_test = get_normalized_data()
    T_train = T_indicator(t_train)
    T_test = T_indicator(t_test)
    
    # Dimensionnality and hyperparameters
    N, D = X_train.shape
    M = 300
    K = 10
    print('Dimensionality: N = {}\t D = {}\t M = {}\t K = {}'.format(N, D, M, K))
    batch_sz = 500
    n_batches = N // batch_sz
    lr = 0.00004
    reg = 0.01
    max_iter = 30 # make it 20 for relu
    mu = 0.9
    print('Hyperparameters: lr = {}\t reg = {}\t velocity = {}\t nb_batches = {}\t batch_size={}\t nb_epochs={}'.format(lr, reg, mu, n_batches, batch_sz, max_iter))
    print_period = 50
    
    # Weights
    W0 = np.random.randn(D, M) / np.sqrt(D)
    b0 = np.zeros(M)
    W1 = np.random.randn(M, K) / np.sqrt(M)
    b1 = np.zeros(K)
    # save initial weights
    W0_0 = W0.copy()
    b0_0 = b0.copy()
    W1_0 = W1.copy()
    b1_0 = b1.copy()

    # 1. Batch
    print('BATCH GRADIENT DESCENT')
    t0 = datetime.now()
    losses_batch = []
    errors_batch = []
    for epoch in range(max_iter):
        for batch_index in range(n_batches):
            X_train_batch = X_train[batch_index*batch_sz:(batch_index*batch_sz + batch_sz),]
            T_train_batch = T_train[batch_index*batch_sz:(batch_index*batch_sz + batch_sz),]
            A1, Y_train_batch = forward(X_train_batch, W0, b0, W1, b1)

            W1 -= lr*(J_derivative_W1(T_train_batch, Y_train_batch, A1) + reg*W1)
            b1 -= lr*(J_derivative_b1(T_train_batch, Y_train_batch) + reg*b1)
            W0 -= lr*(J_derivative_W0(T_train_batch, Y_train_batch, W1, A1, X_train_batch) + reg*W0)
            b0 -= lr*(J_derivative_b0(T_train_batch, Y_train_batch, W1, A1) + reg*b0)

            if batch_index % print_period == 0:
                _ , Y_test = forward(X_test, W0, b0, W1, b1)
                j_test = J(T_test, Y_test)
                losses_batch.append(j_test)
                e = accuracy(predict(Y_test), t_test)
                errors_batch.append(e)
                print("Cost at iteration epoch={}, batch_index={}: {}\t Accuracy = {}".format(epoch, batch_index, round(j_test, 6), e))

    _, Y_test = forward(X_test, W0, b0, W1, b1)
    print("Final accuracy: {}\n".format(accuracy(predict(Y_test), t_test)))
    print("Elapsted time for batch GD: {}\n".format(datetime.now() - t0))
    
    # 2. Batch with momentum
    print('BATCH GRAGIENT DESCENT WITH MOMENTUM')
    t0 = datetime.now()
    
    W0 = W0_0.copy()
    b0 = b0_0.copy()
    W1 = W1_0.copy()
    b1 = b1_0.copy()
    losses_momentum = []
    errors_momentum = []
    
    dW1 = 0
    db1 = 0
    dW0 = 0
    db0 = 0
    
    for epoch in range(max_iter):
        for batch_index in range(n_batches):
            X_train_batch = X_train[batch_index*batch_sz:(batch_index*batch_sz + batch_sz),]
            T_train_batch = T_train[batch_index*batch_sz:(batch_index*batch_sz + batch_sz),]
            A1, Y_train_batch = forward(X_train_batch, W0, b0, W1, b1)

            # gradients
            gW1 = J_derivative_W1(T_train_batch, Y_train_batch, A1) + reg*W1
            gb1 = J_derivative_b1(T_train_batch, Y_train_batch) + reg*b1
            gW0 = J_derivative_W0(T_train_batch, Y_train_batch, W1, A1, X_train_batch) + reg*W0
            gb0 = J_derivative_b0(T_train_batch, Y_train_batch, W1, A1) + reg*b0
            
            # update velocities
            dW1 = mu*dW1 - lr*gW1
            db1 = mu*db1 - lr*gb1
            dW0 = mu*dW0 - lr*gW0
            db0 = mu*db0 - lr*gb0

            # updates
            W1 += dW1
            b1 += db1
            W0 += dW0
            b0 += db0

            if batch_index % print_period == 0:
                _, Y_test = forward(X_test, W0, b0, W1, b1)
                j_test = J(T_test, Y_test)
                losses_momentum.append(j_test)
                e = accuracy(predict(Y_test), t_test)
                errors_momentum.append(e)
                print("Cost at iteration epoch={}, batch_index={}: {}\tAccuracy: {}".format(epoch, batch_index, round(j_test, 6), e))
    _, Y_test_final = forward(X_test, W0, b0, W1, b1)
    print("Final accuracy:", accuracy(predict(Y_test_final), t_test))
    print("Elapsted time for batch GD with Momentum: {}\n".format(datetime.now() - t0))

    # 3. Batch with Nesterov momentum
    print('BATCH GRADIENT DESCENT WITH NESTEROV MOMENTUM')
    t0 = datetime.now()
    W0 = W0_0.copy()
    b0 = b0_0.copy()
    W1 = W1_0.copy()
    b1 = b1_0.copy()

    losses_nesterov = []
    errors_nesterov = []

    vW1 = 0
    vb1 = 0
    vW0 = 0
    vb0 = 0
    
    for epoch in range(max_iter):
        for batch_index in range(n_batches):
            X_train_batch = X_train[batch_index*batch_sz:(batch_index*batch_sz + batch_sz),]
            T_train_batch = T_train[batch_index*batch_sz:(batch_index*batch_sz + batch_sz),]
            A1, Y_train_batch = forward(X_train_batch, W0, b0, W1, b1)

            # updates
            gW1 = J_derivative_W1(T_train_batch, Y_train_batch, A1) + reg*W1
            gb1 = J_derivative_b1(T_train_batch, Y_train_batch) + reg*b1
            gW0 = J_derivative_W0(T_train_batch, Y_train_batch, W1, A1, X_train_batch) + reg*W0
            gb0 = J_derivative_b0(T_train_batch, Y_train_batch, W1, A1) + reg*b0

            # v update
            vW1 = mu*vW1 - lr*gW1
            vb1 = mu*vb1 - lr*gb1
            vW0 = mu*vW0 - lr*gW0
            vb0 = mu*vb0 - lr*gb0

            # param update
            W1 += mu*vW1 - lr*gW1
            b1 += mu*vb1 - lr*gb1
            W0 += mu*vW0 - lr*gW0
            b0 += mu*vb0 - lr*gb0

            if (batch_index % print_period == 0):
                _, Y_test = forward(X_test, W0, b0, W1, b1)
                j_test = J(T_test, Y_test)
                losses_nesterov.append(j_test)
                e = accuracy(predict(Y_test), t_test)
                errors_nesterov.append(e)
                print("Cost at iteration epoch={}, batch_index={}: {}\tAccuracy: {}".format(epoch, batch_index, round(j_test, 6), e))
    _, Y_test_final = forward(X_test, W0, b0, W1, b1)
    print("Final accuracy:", accuracy(predict(Y_test_final), t_test))
    print("Elapsted time for batch GD with Nesterov Momentum: {}\n".format(datetime.now() - t0))
    
    plt.plot(losses_batch, label="batch")
    plt.plot(losses_momentum, label="momentum")
    plt.plot(losses_nesterov, label="nesterov")
    plt.legend()
    plt.show()
    plt.savefig('momentums_relu_activation.png')

if __name__ == '__main__':
    main()
