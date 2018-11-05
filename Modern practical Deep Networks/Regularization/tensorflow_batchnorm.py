import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from util import get_normalized_data

def init_weights(M1, M2):
    return np.random.randn(M1, M2) * np.sqrt(2.0/M1)

class HiddenLayerBatchNorm:
    def __init__(self, M1, M2, f):
        self.M1 = M1
        self.M2 = M2
        self.f = f
        W = init_weights(M1, M2).astype(np.float32)
        gamma = np.ones(M2).astype(np.float32)
        beta = np.zeros(M2).astype(np.float32)
        # No need for bias: it is redundant while applying batch normalization
        self.W = tf.Variable(W)
        self.gamma = tf.Variable(gamma)
        self.beta = tf.Variable(beta)
        # For test time, you need to apply exponentially-smoothed average on
        # the var and mean of all the layers average and var
        # trainable = False, not to apply gradient descent on these shared variables
        self.running_mean = tf.Variable(np.zeros(M2).astype(np.float32), trainable = False)
        self.running_var = tf.Variable(np.zeros(M2).astype(np.float32), trainable = False)
    
    def forward(self, X, is_training, decay=0.9):
        Z = tf.matmul(X, self.W)
        if is_training:
            batch_mean, batch_var = tf.nn.moments(Z, [0])
            update_running_mean = tf.assign(
                    self.running_mean,
                    decay * self.running_mean + (1 - decay) * batch_mean
                    )
            update_running_var = tf.assign(
                    self.running_var,
                    decay * self.running_var + (1 - decay) * batch_var
                    )
            # Pay attention not to interfere between running updates and batches updates
            with tf.control_dependencies([update_running_mean, update_running_var]):
                Z_hat = tf.nn.batch_normalization(
                        Z,
                        batch_mean,
                        batch_var,
                        self.beta,
                        self.gamma,
                        1e-14
                        )
        else:
            Z_hat = tf.nn.batch_normalization(
                    Z,
                    self.running_mean,
                    self.running_var,
                    self.beta,
                    self.gamma,
                    1e-14
                    )
        return self.f(Z_hat)

# Class built for the output layer: tensorflow apply softmax during training 
# so no activation function 
class HiddenLayer:
    def __init__(self, M1, M2, f):
        self.M1 = M1
        self.M2 = M2
        self.f = f
        W = init_weights(M1, M2)
        b = np.zeros(M2)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))

    def forward(self, A):
        return self.f(tf.matmul(A, self.W) + self.b)

class ANN:
    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes

    def set_session(self, session):
        self.session = session

    def fit(self, X_train, X_test, t_train, t_test, activation=tf.nn.relu, lr=1e-2, epochs=15, batch_size=100, print_period=100, show_fig=True):
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        t_train = t_train.astype(np.int64)
        t_test = t_test.astype(np.int64)
        
        # Dimensionality
        N, D = X_train.shape
        K = len(set(t_train))
        nb_batches = N // batch_size
        print('Dimensionality: N:{} batch_size:{} nb_batches:{} epochs:{} -- D:{} hidden:{} K:{}'.format(N, batch_size, nb_batches, epochs, D, self.hidden_layer_sizes, K))
        
        ### Initialization
        # Hidden layers initialization
        self.layers = []
        M1 = D
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayerBatchNorm(M1, M2, activation)
            self.layers.append(h)
            M1 = M2
        # Output layer initialization
        h = HiddenLayer(M1, K, lambda x: x)
        self.layers.append(h)
        
        ### Tf general setups: placeholders
        X_pl = tf.placeholder(tf.float32, shape=(None, D), name='X')
        t_pl = tf.placeholder(tf.int32, shape=(None,), name='t')
        # FOR LATER USE
        self.X_pl = X_pl

        ### Tf training setups
        Y_train = self.forward(X_pl, True)
        J_train = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits = Y_train,
                    labels = t_pl
                    )
                )
        train_opt = tf.train.MomentumOptimizer(lr, momentum = 0.9, use_nesterov=True).minimize(J_train)
        
        ### Tf test setups
        Y_test = self.forward(X_pl, False)
        self.y_test = tf.argmax(Y_test, 1)
        
        ### Tf varaiable initialization
        self.session.run(tf.global_variables_initializer())

        ### TRAINING & TEST
        costs_train = []
        for epoch in range(epochs):
            if nb_batches > 1:
                X_train, t_train = shuffle(X_train, t_train)
            for batch_index in range(nb_batches):
                X_batch = X_train[batch_index*batch_size:(batch_index+1)*batch_size,]
                t_batch = t_train[batch_index*batch_size:(batch_index+1)*batch_size,]
                j_train, _, Y_bn = self.session.run([J_train, train_opt, Y_train], feed_dict={X_pl:X_batch, t_pl:t_batch})
                costs_train.append(j_train)
                if (batch_index+1) % print_period == 0:
                    acc = np.mean(t_batch == np.argmax(Y_bn, 1))
                    print('Epoch: {}, batch_indx: {} -- training cost: {} training accuracy: {}'.format(epoch, batch_index+1, j_train, acc))
            print('Epoch {} - Training accuracy: {}% - Test accuracy:{}%'.format(epoch, self.final_acc(X_train, t_train) * 100, self.final_acc(X_test, t_test) * 100))
        print('Final TRAINING accuracy: {}'.format(self.final_acc(X_train, t_train)))
        print('Final TEST accuracy: {}'.format(self.final_acc(X_test, t_test)))
        if show_fig:
            plt.plot(costs_train)
            plt.show()

    def forward(self, X, is_training):
        A = X
        # Hidden layers
        for h in self.layers[:-1]:
            A = h.forward(A, is_training)
        # Output layer
        Y = self.layers[-1].forward(A)
        return Y

    def y(self, X):
        return self.session.run(self.y_test, feed_dict={self.X_pl : X})

    def final_acc(self, X, t):
        y = self.y(X)
        return np.mean(t == y)


def main():
    X_train, X_test, t_train, t_test = get_normalized_data()
    ann = ANN([500,300])
    session = ann.set_session(tf.InteractiveSession())
    ann.fit(X_train, X_test, t_train, t_test, show_fig = True)
    writer = tf.summary('./tensorboard_logs/demo1')
    writer.add_graph(session.graph)

if __name__ == '__main__':
    main()

                    
