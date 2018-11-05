import numpy as np
import sys

N = input('How many samples are you training on ? \n --> ')
k = input('How many classes contains your output layer ? \n --> ')
# Here the output of the output layer is a considered as a (k,N)-shaped array of floating-point samples from the standard normal distribution
A = np.random.randn(int(N), int(k))
print('A = \n{}'.format(A))
# Expentionate the matrix to get positive scores 
expA = np.exp(A)
print('exp(A) = \n{}'.format(expA))
# Get a probability distribution usinge SOFTMAX
answer = expA / expA.sum(axis=1, keepdims = True)
print('softmax(A) = \n{}'.format(answer))
print('expA.sum(axis = 1, keepdims = True) = \n {} '.format(expA.sum(axis=1, keepdims = True)))
# Sum of the probability = 1
print(answer.sum())
