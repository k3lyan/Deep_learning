import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from datetime import datetime

from benchmark import get_data, accuracy 

def convpool(X, W, b):
    conv_out = tf.nn.conv2d(X, W, strides=[1,1,1,1], padding='SAME')
    conv_out = tf.nn.bias_add(conv_out, b)
    pool_out = tf.nn.max_pool(conv_out, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding='SAME')
    return tf.nn.relu(pool_out)

def init_filter(shape, pool_size):
    #w = np.random.randn(*shape) * np.sqrt(2) / np.sqrt(np.prod(shape[:-1]) + shape[-1]* np.prod(shape[:-2])/np.prod(pool_size))
    w = np.random.randn(*shape) * np.sqrt(2.0 / np.prod(shape[:-1]))
    return w.astype(np.float32)

def rearrange(X):
    # input is (32,32,3,N)
    # output is (N, 32, 32, 3)
    # N = X.shape[-1]
    # out = np.zeros((N, 32, 32, 3), dtype=np.float32)
    # for i in range(N):
    #   for j in range(3):
    #       out[i,:,:,j] = X[:,:,i,j]
    #return out / 255
    return (X.transpose(3,0,1,2)/255).astype(np.float32)

def main():
    train, test = get_data()
    
    X_train = rearrange(train['X'])
    t_train = train['y'].flatten() - 1
    del train 
    X_train, t_train = shuffle(X_train, t_train)
    X_test = rearrange(test['X'])
    t_test = test['y'].flatten() - 1
    del test

    # Gradient-descent parameters
    epochs = 6
    print_period = 10
    N = X_train.shape[0]
    batch_size = 500
    nb_batches = N // batch_size
    
    # Limit samples since input will always have to be same size
    # we could have done: N = N / batch_size * batch_size
    X_train = X_train[:73000,]
    t_train = t_train[:73000]
    X_test = X_test[:26000,]
    t_test = t_test[:26000]
    
    # Initial weights
    M = 500
    K = 10
    pool_size = (2,2)
    # W*H*C1*features_map
    W0_shape = (5, 5, 3, 20)
    W0_init = init_filter(W0_shape, pool_size)
    b0_init = np.zeros(W0_shape[-1], dtype=np.float32)

    W1_shape = (5, 5, 20, 50)
    W1_init = init_filter(W1_shape, pool_size)
    b1_init = np.zeros(W1_shape[-1], dtype=np.float32)
    # ANN weights
    W2_init = np.random.randn(W1_shape[-1]*8*8, M) / np.sqrt(W1_shape[-1]*8*8+M)
    b2_init = np.zeros(M)
    W3_init = np.random.randn(M, K) / np.sqrt(M+K)
    b3_init = np.zeros(K)
    
    # tf environment
    X_pl = tf.placeholder(tf.float32, shape=(batch_size, 32, 32, 3), name='X')
    t_pl = tf.placeholder(tf.int32, shape=(batch_size,), name='t')
    W0 = tf.Variable(W0_init.astype(np.float32))
    b0 = tf.Variable(b0_init.astype(np.float32))
    W1 = tf.Variable(W1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))
    W3 = tf.Variable(W3_init.astype(np.float32))
    b3 = tf.Variable(b3_init.astype(np.float32))
    
    # tf training environment
    A1 = convpool(X_pl, W0, b0)
    A2 = convpool(A1, W1, b1)
    A2_shape = A2.get_shape().as_list()
    A2r = tf.reshape(A2, [A2_shape[0], np.prod(A2.shape[1:])])
    A3 = tf.nn.relu(tf.matmul(A2r, W2) + b2)
    Z4 = tf.matmul(A3, W3) + b3 
    J = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels = t_pl,
                logits = Z4
                )
            ) 
    train_op = tf.train.RMSPropOptimizer(0.0001, decay=0.99, momentum=0.9).minimize(J)

    # tf test environment
    y = tf.argmax(Z4, 1)
    
    # TRAIN & TEST
    t0 = datetime.now()
    tests_costs = []
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs):
            for batch_id in range(nb_batches):
                X_train_batch = X_train[batch_id*batch_size:(batch_id+1)*batch_size,]
                t_train_batch = t_train[batch_id*batch_size:(batch_id+1)*batch_size,]
                if len(X_train_batch) == batch_size:
                    sess.run(train_op, feed_dict={X_pl:X_train_batch, t_pl:t_train_batch})
                    if batch_id % print_period == 0:
                        # due to RAM limitations we need to have a fixed input
                        # We took the size of a batch for the placeholder
                        # as a result we have this ugly total cost and prediction computation
                        j_test = 0
                        y_test = np.zeros(len(X_test))
                        for batch_test_id in range(len(X_test)//batch_size):
                            X_test_batch = X_test[batch_test_id*batch_size:(batch_test_id+1)*batch_size,]
                            t_test_batch = t_test[batch_test_id*batch_size:(batch_test_id+1)*batch_size]
                            j_test += sess.run(J, feed_dict={X_pl:X_test_batch, t_pl:t_test_batch})
                            y_test[batch_test_id*batch_size:(batch_test_id+1)*batch_size,] = sess.run(
                                    y,
                                    feed_dict={X_pl:X_test_batch}
                                    )
                        tests_costs.append(j_test)
                        acc = accuracy(y_test, t_test)
                        print('Epoch {} batch_id {}: validation cost: {} - accuracy = {}%'.format(epoch, batch_id, j_test, acc*100))
       # W0_val = W0.eval()
       # W1_val = W1.eval()
    print('Elapsed time: {}'.format(datetime.now()-t0))
    #plt.plot(tests_costs)
    #plt.show()
    '''
    # SHOW WHAT THE MODEL LEARNED THROUGH CONV
    W0_val = W0_val.transpose(3, 2, 0, 1)
    W1_val = W1_val.transpose(3, 2, 0, 1)
    
    # visualize the first filter W0 (20, 3, 5, 5)
    # 20*3 = 60 ~ 64 = 8*8 we let the last 
    # four squares empty
    # image: 5*5
    grid = np.zeros((8*5, 8*5))
    m = 0
    n = 0
    for i in range(20):
        for j in range(3):
            filt = W0_val[i,j]
            grid[m*5:(m+1)*5,n*5:(n+1)*5] = filt
            m += 1
            if m >= 8:
                m = 0
                n += 1
    plt.imshow(grid, cmap='gray')
    plt.title("W0")
    plt.show()

    # visualize the second filter W1 (50, 20, 5, 5)
    # same as before: 
    # 20*50 = 1000 ~ 32**2 = 1024
    grid = np.zeros((32*5, 32*5))
    m = 0
    n = 0
    for i in range(50):
        for j in range(20):
            filt = W2_val[i,j]
            grid[m*5:(m+1)*5,n*5:(n+1)*5] = filt
            m += 1
            if m >= 32:
                m = 0
                n += 1
   plt.imshow(grid, cmap='gray')
    plt.title("W1")
    plt.show()
   ''' 
if __name__ == '__main__':
    main()
