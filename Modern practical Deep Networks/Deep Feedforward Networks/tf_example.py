from __future__ import print_function, division
from builtins import range
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# nb of inputs by class
N_class = 500

# Dimensions of the NN
D = 2
M = 3
K = 3

# Inputs
X1 = np.random.randn(N_class, D) + np.array([0, -2])
X2 = np.random.randn(N_class, D) + np.array([2, 2])
X3 = np.random.randn(N_class, D) + np.array([-2, 2])
X = np.vstack([X1, X2, X3]).astype(np.float32)

# targets
t = np.array([0]*N_class + [1]*N_class + [2]*N_class)
N = len(t)
T = np.zeros((N, K))
for n in range(N):
    T[n, t[n]] = 1
'''
def T_indicator(t, K):
    N = len(t)
    T = np.zeros((N, K))
    for n in range(N):
        T[n, t[n]] = 1
T = T_indicator(t, K)        
'''

# Have a look at the inputs
plt.scatter(X[:,0], X[:,1], c=t, s=100, alpha=0.5)
#plt.show()

# Initialize the weights with tf
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# Forward with tf
# tf != numpy: we return NOT the softmax but JUST the activation summated !!
def forward(X, W0, b0, W1, b1):
    Z = tf.matmul(X, W0) + b0
    A1 = tf.nn.sigmoid(Z)
    return tf.matmul(A1, W1) + b1

tfX = tf.placeholder(tf.float32, [None, D])
tfT = tf.placeholder(tf.float32, [None, K])

W0 = init_weights([D, M])
b0 = init_weights([M])
W1 = init_weights([M, K])
b1 = init_weights([K])

# Y = P(y=k|X, parameters)
Y = forward(tfX, W0, b0, W1, b1)

# tf is going to calculate the gradients and apply gradient-descent automatically !
# compute costs
# WARNING: This op expects unscaled logits,
# since it performs a softmax on logits
# internally for efficiency.
# Do not call this op with the output of softmax,
# as it will produce incorrect results.
cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tfT,
            logits=Y
            )
        )

train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
# argmax of the output on axis=1
predict_op = tf.argmax(Y, 1) 

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for epoch in range(1000):
    sess.run(train_op, feed_dict={tfX: X, tfT: T})
    pred = sess.run(predict_op, feed_dict={tfX: X, tfT: T})
    if (epoch % 100 == 0):
        print(np.mean(t == pred))

