import numpy as np
import theano
import theano.tensor as Tens
import matplotlib.pyplot as plt

from util import get_normalized_data, T_indicator, accuracy

def relu(z):
    return z * (z > 0)

def main():
    # STEP 1 : get the data and define all the usual variables
    X_train, X_test, t_train, t_test = get_normalized_data()
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    t_train = t_train.astype(np.float32)
    t_test = t_test.astype(np.float32)
    T_train = T_indicator(t_train).astype(np.float32)
    T_test = T_indicator(t_test).astype(np.float32)
    
    # Dimensionality
    N, D = X_train.shape
    M = 300
    K = 10
    max_iter = 20
    print_period = 10
    batch_size = 500
    nb_batches = N // batch_size

    # Hyperparameters
    lr = 0.0004
    reg = 0.01
     
    # Initialize weights
    W0_init = np.random.randn(D, M) / np.sqrt(D)
    b0_init = np.zeros(M)
    W1_init = np.random.randn(M, K) / np.sqrt(M)
    b1_init = np.zeros(K)

    # STEP 2 : define Theano variables and expressions
    # data and labels to go into the inputs 
    thX = Tens.matrix('X')
    thT = Tens.matrix('T')
    # weights to be update
    W0 = theano.shared(W0_init, 'W0')
    b0 = theano.shared(b0_init, 'b0')
    W1 = theano.shared(W1_init, 'W1')
    b1 = theano.shared(b1_init, 'b1')
    # use the built-in theano functions to forward
    thA = relu(thX.dot(W0) + b0)
    thY = Tens.nnet.softmax(thA.dot(W1) + b1)
    # define cost function
    J = -(thT * Tens.log(thY)).sum() + reg*((W1*W1).sum() + (b1*b1).sum() + (W0*W0).sum() + (b0*b0).sum())
    prediction = Tens.argmax(thY, axis=1)
    
    # STEP 3 : training expression and function
    update_W1 = W1 - lr * Tens.grad(J, W1)
    update_b1 = b1 - lr * Tens.grad(J, b1)
    update_W0 = W0 - lr * Tens.grad(J, W0)
    update_b0 = b0 - lr * Tens.grad(J, b0)
    
    train = theano.function(
                inputs = [thX, thT],
                updates = [(W1, update_W1), (b1, update_b1), (W0, update_W0), (b0, update_b0)],
            )

    # the prediction over the whole dataset
    prediction_state = theano.function(
                inputs= [thX, thT],
                outputs = [J, prediction]
            )
    
    Js = []
    for epoch in range(max_iter):
        for batch_index in range(nb_batches):
            X_batch = X_train[batch_index*batch_size:(batch_index+1)*batch_size,]
            T_batch = T_train[batch_index*batch_size:(batch_index+1)*batch_size,]
            train(X_batch, T_batch)

            if batch_index % print_period == 0:
                J_test, prediction_test = prediction_state(X_test, T_test)
                Js.append(J_test)
                print('Epoch {}\t batch_index {}:\t J {}\t accuracy {}'.format(epoch, batch_index, J_test, accuracy(prediction_test, t_test)))
    plt.plot(Js)
    plt.savefig('cost_theano_nnet.png')

if __name__ == '__main__':
    main()

