import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from util import get_normalized_data, accuracy
from sklearn.utils import shuffle

class HiddenLayer:
    def __init__(self, M1, M2):
        self.M1 = M1
        self.M2 = M2
        W = np.random.randn(M1, M2) / np.sqrt(2.0 / M2)
        b = np.zeros(M2)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))
        self.params = [self.W, self.b]

    def forward(self, X):
        return tf.nn.relu(tf.matmul(X, self.W) + self.b)

class ANN:
    def __init__(self, hidden_layer_sizes, p_keep):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_ratio = p_keep

    def fit(self, X_train, X_test, t_train, t_test, lr=1e-4, mu=0.9, decay=0.9, epochs = 30, batch_size=100, print_period=50):
        
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        t_train = t_train.astype(np.int64)
        t_test = t_test.astype(np.int64)
        # Dimensionality
        N, D = X_train.shape
        K = len(set(t_train))
        nb_batches = N // batch_size
        print('Dimensionnality: N: {}, D:{}, M1:{}, M2:{}, K:{}, nb_batches: {}, batch_size:{}'.format(N, D, self.hidden_layer_sizes[0], self.hidden_layer_sizes[1], K, nb_batches, batch_size))
        print('Hyperparameters: lr:{} mu:{} decay:{} epochs:{}'.format(lr, mu, decay, epochs))
        # hidden layer (+ input layer)
        self.hidden_layers = []
        M1 = D
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2)
            self.hidden_layers.append(h)
            M1 = M2
        # output layer
        W = np.random.randn(M1, K) / np.sqrt(2.0/M1)
        b = np.zeros(K)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))
        # Save the parameters !!
        self.params = [self.W, self.b]
        for h in self.hidden_layers:
            self.params += h.params
        
        ######## SET UP THE PLACEHOLDERS
        X_pl = tf.placeholder(tf.float32, shape=(None, D), name='inputs') 
        t_pl = tf.placeholder(tf.int64, shape=(None,), name='labels')
        
        ####### SET UP THE TRAINING ENVIRONMENT
        Y_train = self.forward_train(X_pl) # logits only, softmax is handled in the cost function
        J = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits = Y_train,
                        labels = t_pl
                    )
                )
        train_op = tf.train.RMSPropOptimizer(lr, decay=decay, momentum=mu).minimize(J)
        # train_op = tf.train.MomentumOptimizer(lr, momentum=mu).minimize(cost)
        # train_op = tf.train.AdamOptimizer(lr).minimize(cost)

        ####### SET UP THE TEST ENVIRONMENT
        Y_test = self.forward_test(X_pl)
        J_test = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits= Y_test,
                    labels = t_pl
                    )
                )
        y = self.predict(X_pl)
        
        #### INITIALIZE TF GLOBAL VARIABLES
        init = tf.global_variables_initializer()

        ####### TRAIN & TEST
        costs_test = []
        with tf.Session() as session:
            session.run(init)
            for epoch in range(epochs):
                X_train, t_train = shuffle(X_train, t_train)
                for batch_index in range(nb_batches):
                    X_batch = X_train[batch_index * batch_size: (batch_index+1) * batch_size]
                    t_batch = t_train[batch_index * batch_size: (batch_index+1) * batch_size]
                    session.run(train_op, feed_dict={X_pl: X_batch, t_pl: t_batch})

                    if batch_index % print_period == 0:
                        j = session.run(J_test, feed_dict={X_pl: X_test, t_pl: t_test})
                        pred = session.run(y, feed_dict={X_pl: X_test})
                        acc = accuracy(pred, t_test)
                        costs_test.append(j)
                        print('Epoch: {}\t batch_indx: {}\t cost: {}\t accuracy:{}'.format(epoch, batch_index, j, acc))
        plt.plot(costs_test)
        plt.show()

    def forward_train(self, X_train):
        # tf.nn.dropout scales inputs by 1/p_keep
        # therefore during training we don't have to scale anything
        A = X_train
        A = tf.nn.dropout(A, self.dropout_ratio[0])
        for h, p_keep in zip(self.hidden_layers, self.dropout_ratio[1:]):
            A = h.forward(A)
            A = tf.nn.dropout(A, p_keep)
        return tf.matmul(A, self.W) + self.b

    def forward_test(self, X_test):
        A = X_test
        for h in self.hidden_layers:
            A = h.forward(A)
        return tf.matmul(A, self.W) + self.b

    def predict(self, X_test):
        Y_test = self.forward_test(X_test)
        return tf.argmax(Y_test, 1)

def main():
    X_train, X_test, t_train, t_test = get_normalized_data()
    ann = ANN([500, 300], [0.8, 0.54, 0.5])
    ann.fit(X_train, X_test, t_train, t_test)

if __name__ == '__main__':
    main()

