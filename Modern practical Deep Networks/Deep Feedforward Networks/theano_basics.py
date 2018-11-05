import theano.tensor as Tens

# Different types of variables
c = Tens.scalar('c')
v = Tens.vector('v')
A = Tens.matrix('A')

# We can define a matrix multiplication
v2 = A.dot(v)

import theano

matrix_times_vector = theano.function(inputs= [A,v], outputs=v2)

import numpy as np
A_val = np.array([[1,2],[3,4]])
v_val = np.array([5, 6])

v2_val = matrix_times_vector(A_val, v_val)
print('{} . {} = {}'.format(A_val, np.transpose(v_val), np.transpose(v2_val)))

# To apply gradient descent, we need to create a SHARED VARIABLE
# This add another level of complexity to the theano function
# 1st arg: initial value, 2nd arg: name 
theta = theano.shared(20.0, 'x')

# cost function with a minimal value
cost = theta*theta + theta + 1

# in Theano you don't have to compute gradients yourself
theta_update = theta - 0.3*Tens.grad(cost, theta)

# in later examples, data and labels will go into the inputs
# and model params will go in the updates
# updates takes in a list of tuples, each tuple has 2 things in it:
# 1) the shared variable to update, 2) the update expression
train = theano.function(inputs=[], outputs=cost, updates=[(theta, theta_update)])

for epoch in range(25):
    cost_val = train()
    print('Epoch {} cost: {}'.format(epoch, cost_val))

print('Optimal value for theta: {}'.format(theta.get_value()))

