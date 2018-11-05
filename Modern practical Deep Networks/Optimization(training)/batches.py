# In this file we compare the progression of the cost function vs. iteration
# for 3 cases:
# 1) full gradient descent
# 2) batch gradient descent
# 3) stochastic gradient descent
#
# We use the PCA-transformed data to keep the dimensionality down (D=300)
# I've tailored this example so that the training time for each is feasible.
# So what we are really comparing is how quickly each type of GD can converge,
# (but not actually waiting for convergence) and what the cost looks like at
# each iteration.
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime

from util import get_pca_normalized_data, forward, accuracy, J, gradW, gradb, T_indicator, predict

# lr = sys.argv[1], ex: 0.0001
# reg = sys.argv[2], ex: 0.01 
# batch_size = sys.argv[3], ex: 500

def main():
    X_train, X_test, t_train, t_test = get_pca_normalized_data()
    print("Performing multi-class logistic regression...\n")

    N, D = X_train.shape
    K = 10
    T_train = T_indicator(t_train)
    T_test = T_indicator(t_test)
    
    lr = float(sys.argv[1])
    reg = float(sys.argv[2])
    batch_size = int(sys.argv[3])

    ######## 1. FULL GRADIENT DESCENT ######## 
    print('Full Gradient Descent')
    W = np.random.randn(D, K) / np.sqrt(D)
    b = np.zeros(K)
    J_test_full = []
    t0 = datetime.now()
    for epoch in range(50):
        Y_train = forward(X_train, W, b)
        W -= lr*(gradW(T_train, Y_train, X_train) - reg*W)
        b -= lr*(gradb(T_train, Y_train) - reg*b)
        
        Y_test = forward(X_test, W, b)
        j_test = J(T_test, Y_test)
        J_test_full.append(j_test)

        if epoch % 1 == 0:
            err = accuracy(predict(Y_test), t_test)
            if epoch % 10 == 0:
                print("Epoch {}:\tcost: {}\taccuracy: {}".format(epoch, round(j_test, 4), err))
    Y_test = forward(X_test, W, b)
    print("Final accuracy:", accuracy(predict(Y_test), t_test))
    print("Elapsted time for full GD: {}\n".format(datetime.now() - t0))

    ######## 2. STOCHASTIC GRADIENT DESCENT ######## 
    print('Stochastic Gradient Descent')
    W = np.random.randn(D,K) / np.sqrt(D)
    b = np.zeros(K)
    J_test_stochastic = []
    t0 = datetime.now()
    for epoch in range(50): # takes very long since we're computing cost for 41k samples
        tmpX, tmpT = shuffle(X_train, T_train)
        for n in range(min(N, 500)): # shortcut so it won't take so long...
            x = tmpX[n,:].reshape(1,D)
            t = tmpT[n,:].reshape(1,10)
            Y_train = forward(x, W, b)

            W -= lr*(gradW(t, Y_train, x) - reg*W)
            b -= lr*(gradb(t, Y_train) - reg*b)

            Y_test = forward(X_test, W, b)
            j_test = J(T_test, Y_test)
            J_test_stochastic.append(j_test)

        if epoch % 1 == 0:
            err = accuracy(predict(Y_test), t_test)
            if epoch % 10 == 0:
                print("Epoch {}:\tcost: {}\taccuracy: {}".format(epoch, round(j_test, 4), err))
    Y_test_final = forward(X_test, W, b)
    print("Final accuracy:", accuracy(predict(Y_test_final), t_test))
    print("Elapsted time for SGD: {}\n".format(datetime.now() - t0))

    ######## 3. BATCH GRADIENT DESCENT ######## 
    print('Batch Gradient Descent')
    W = np.random.randn(D, K) / np.sqrt(D)
    b = np.zeros(K)
    J_test_batch = []
    nb_batches = N // batch_size
    t0 = datetime.now()
    for epoch in range(50):
        tmpX, tmpT = shuffle(X_train, T_train)
        for batch_index in range(nb_batches):
            x = tmpX[batch_index*batch_size:(batch_index*batch_size + batch_size),:]
            t = tmpT[batch_index*batch_size:(batch_index*batch_size + batch_size),:]
            Y_train = forward(x, W, b)

            W -= lr*(gradW(t, Y_train, x) - reg*W)
            b -= lr*(gradb(t, Y_train) - reg*b)

            Y_test = forward(X_test, W, b)
            j_test = J(T_test, Y_test)
            J_test_batch.append(j_test)
        if epoch % 1 == 0:
            err = accuracy(predict(Y_test), t_test)
            if epoch % 10 == 0:
                print("Epoch {}\tcost: {}\taccuracy: {}".format(epoch, round(j_test, 4), err))
    Y_test_final = forward(X_test, W, b)
    print("Final accuracy:", accuracy(predict(Y_test_final), t_test))
    print("Elapsted time for batch GD:", datetime.now() - t0)

    ######## PLOTS ########  
    x1 = np.linspace(0, 1, len(J_test_full))
    plt.plot(x1, J_test_full, label="full")
    x2 = np.linspace(0, 1, len(J_test_stochastic))
    plt.plot(x2, J_test_stochastic, label="stochastic")
    x3 = np.linspace(0, 1, len(J_test_batch))
    plt.plot(x3, J_test_batch, label="batch")
    plt.legend()
    #plt.savefig('full_vs_stoch_vs_batch_lr={}_reg={}_batch_size={}.png'.format(lr, reg, batch_size))
    plt.show()

if __name__ == '__main__':
    main()
