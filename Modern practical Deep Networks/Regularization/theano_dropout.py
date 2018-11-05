# A 1-hidden-layer neural network in Theano.
# This code is not optimized for speed.
# It's just to get something working dropout principles.

import numpy as np
import theano
import theano.tensor as Tens
import matplotlib.pyplot as plt
from theano.tensor.shared_randomstreams import RandomStreams
from util import get_normalized_data
from sklearn.utils import shuffle

def accuracy(pred, t):
    return np.mean(pred == t)

class HiddenLayer:
    def __init__(self, M1, M2, an_id):
        self.id = an_id
        self.M1 = M1
        self.M2 = M2 
        W = np.random.randn(M1, M2) / np.sqrt(2.0 / M1)
        b = np.zeros(M2)
        self.W = theano.shared(W, 'W_%s' % self.id)
        self.b = theano.shared(b, 'b_%s'% self.id)
        self.params = [self.W, self.b]

    def forward(self, X):
        return Tens.nnet.relu(X.dot(self.W) + self.b) #self.A

class ANN:
    def __init__(self, hidden_layer_sizes, p_keep):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_rates = p_keep
        
    def fit(self, X_train, X_valid, t_train, t_valid, learning_rate=1e-4, mu=0.9, decay=0.9, epochs=8, batch_size=100, show_fig=False):
        
        ###### GET THE INPUTS, TARGETS AND PARAMETERS OF YOUR MODEL
        # Inputs and targets
        X_train = X_train.astype(np.float32)
        X_valid = X_valid.astype(np.float32)
        t_train = t_train.astype(np.int32)
        t_valid = t_valid.astype(np.int32)
        # Random droping during training
        self.rng = RandomStreams()
        # Hidden layers (taking the input layer in consideration)
        N, D = X_train.shape
        K = len(set(t_train))
        self.hidden_layers = []
        M1 = D
        count = 0
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2, count)
            self.hidden_layers.append(h)
            M1 = M2
            count += 1
        # Output layer
        W = np.random.randn(M1,K) * np.sqrt(2.0/M1)
        b = np.zeros(K)
        self.W = theano.shared(W, 'W_out')
        self.b = theano.shared(b, 'b_out')
        # Collect the parameters for later use
        self.params = [self.W, self.b]
        for h in self.hidden_layers:
            self.params += h.params
        print('ANN self.params: {}'.format(self.params))
        

        ####### SET UP THEANO GENERAL VARIABLES (placeholders for inputs and targets)
        thX = Tens.matrix('X')
        tht = Tens.ivector('T')
        
        ####### SET UP THEANO TRAINING VARIABLES & FUNCTIONS
        Y_train = self.forward_train(thX)
        # Training cost
        J = -Tens.mean(Tens.log(Y_train[Tens.arange(tht.shape[0]), tht]))
        # Gradients wrt each param
        grads = Tens.grad(J, self.params)
        # Momentum: initialized to 0
        dparams = [theano.shared(np.zeros_like(p.get_value())) for p in self.params]
        # RMSProp: cache intialized to 1
        cache = [theano.shared(np.ones_like(p.get_value())) for p in self.params]
        new_cache = [decay*c + (1-decay)*g*g for param, c, g in zip(self.params, cache, grads)]
        new_dparams = [mu*dp - learning_rate*g/Tens.sqrt(new_c + 1e-10) for param, new_c, dp, g in zip(self.params, new_cache, dparams, grads)]
        
        updates = [
            (c, new_c) for c, new_c in zip(cache, new_cache)
        ] + [
            (dp, new_dp) for dp, new_dp in zip(dparams, new_dparams)
        ] + [
            (p, p + new_dp) for p, new_dp in zip(self.params, new_dparams)
        ]

        train_op = theano.function(
            inputs = [thX, tht],
            updates = updates
        )


        ####### SET UP THEANO TEST VARIABLES AND FUNCTIONS
        Y_valid = self.forward_predict(thX)
        J_test = -Tens.mean(Tens.log(Y_valid[Tens.arange(tht.shape[0]), tht]))
        predictions = self.predict(thX)
        
        predict_op = theano.function(
                inputs = [thX, tht], 
                outputs = [J_test, predictions]
        )
        
        ###### TRAIN & TEST
        nb_batches = N // batch_size
        costs = []
        for epoch in range(epochs):
            X_train, t_train = shuffle(X_train, t_train)
            for batch_index in range(nb_batches):
                X_batch = X_train[batch_index*batch_size:(batch_index+1)*batch_size]
                t_batch = t_train[batch_index*batch_size:(batch_index+1)*batch_size]
                train_op(X_batch, t_batch)
                if batch_index % 50 == 0:
                    j, pred = predict_op(X_valid, t_valid)
                    costs.append(j)
                    acc = accuracy(t_valid, pred)
                    print("epoch:", epoch, "batch_index:", batch_index, "nb_batches:", nb_batches, "cost:", j, "accuracy:", acc)
        
        if show_fig:
            plt.plot(costs)
            plt.show()
    
    # For the training: apply a random mask to each layers inputs, to randomly drop nodes
    def forward_train(self, X):
        A = X
        for h, p_keep in zip(self.hidden_layers, self.dropout_rates[:-1]):
            mask = self.rng.binomial(n=1, p=p_keep, size=A.shape)
            A = mask * A
            A = h.forward(A)
        mask = self.rng.binomial(n=1, p=self.dropout_rates[-1], size=A.shape)
        A = mask * A
        return Tens.nnet.softmax(A.dot(self.W) + self.b) #softmax at the output layer as usual

    # For the testing: multiply the layer input by p_keep to simulate all the NN possibilies (probabilistic point of view)
    def forward_predict(self, X_test):
        A = X_test
        for h, p_keep in zip(self.hidden_layers, self.dropout_rates[:-1]):
            A = h.forward(p_keep * A)
        return Tens.nnet.softmax((self.dropout_rates[-1] * A).dot(self.W) + self.b)

    def predict(self, X_test):
        Y_test = self.forward_predict(X_test)
        return Tens.argmax(Y_test, axis=1)

def main():
    X_train, X_test, t_train, t_test = get_normalized_data()
    ann = ANN([500, 300], [0.8, 0.5, 0.5])
    ann.fit(X_train, X_test, t_train, t_test, show_fig=True)

if __name__ == '__main__':
    main()
