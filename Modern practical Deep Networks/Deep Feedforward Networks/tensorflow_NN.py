import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from util import get_normalized_data, T_indicator, accuracy

def main():
    ## STEP 1: get the data and define all the usual variables
    X_train, X_test, t_train, t_test = get_normalized_data()
    T_train = T_indicator(t_train)
    T_test = T_indicator(t_test)
    # Dimensionality
    max_iter = 15 
    print_period = 50
    N, D = X_train.shape
    M1 = 300
    M2 = 100
    K = 10
    batch_size = 500
    nb_batches = N // batch_size
    print('Dim: N:{}\t D:{}\t M1:{}\t M2:{}\t K:{}\t batch_size:{}\t nb_batches={}'.format(N, D, M1, M2, K, batch_size, nb_batches))
    # Hyperparameters
    lr = 0.0004
    reg = 0.01
    print('HP: lr:{}\t reg:{}'.format(lr, reg))
    # Weigths initialization
    W0_init = np.random.randn(D, M1) / 28
    b0_init = np.zeros(M1)
    W1_init  = np.random.randn(M1, M2) / np.sqrt(M1)
    b1_init  = np.zeros(M2)
    W2_init  = np.random.randn(M2, K) / np.sqrt(M2)
    b2_init  = np.zeros(K)

    ## STEP 2: DEFINE Tensorflow variables and expressions
    X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    T = tf.placeholder(tf.float32, shape=(None, K), name='T')
    
    W0 = tf.Variable(W0_init.astype(np.float32))
    b0 = tf.Variable(b0_init.astype(np.float32))
    W1 = tf.Variable(W1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))
    
    A1 = tf.nn.relu(tf.matmul(X,W0) + b0)
    A2 = tf.nn.relu(tf.matmul(A1,W1) + b1)
    # U and not Y because softmax is taken care while calculating the cost
    U = tf.matmul(A2,W2) + b2
    J = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=U, labels=T))

    ## STEP 3: GD updates expressions, training and predict functions expressions
    # The optimizeris are already implemented
    # let's go with RMSprop, it includes momentum.
    train_op = tf.train.RMSPropOptimizer(lr, decay=0.99, momentum=0.9).minimize(J)
    predict_op = tf.argmax(U, 1)
    
    ## STEP 4: TRAINING
    costs = []
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        for epoch in range(max_iter):
            for batch_index in range(nb_batches):
                X_batch = X_train[batch_index*batch_size:(batch_index+1)*batch_size,]
                T_batch = T_train[batch_index*batch_size:(batch_index+1)*batch_size,]
                session.run(train_op, feed_dict={X:X_batch, T:T_batch})
                if batch_index % print_period == 0:
                    j = session.run(J, feed_dict={X: X_test, T:T_test})
                    costs.append(j)
                    prediction = session.run(predict_op, feed_dict={X: X_test})
                    acc = accuracy(prediction, t_test)
                    print('Epoch: {}\t cost:{}\t accuracy:{}'.format(epoch, round(j,3), round(acc,3)))
    plt.plot(costs)
    plt.savefig('./experiments/tf_NN_RMSProp_N={}_D={}_M1={}_M2={}_K={}_batch_size={}_nb_batches={}_lr={}_reg={}.png'.format(N, D, M1, M2, K, batch_size, nb_batches, lr, reg))

if __name__ == '__main__':
    main()

