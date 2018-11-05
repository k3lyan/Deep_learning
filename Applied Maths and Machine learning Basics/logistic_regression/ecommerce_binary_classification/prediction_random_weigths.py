import numpy as np
from pre_process import get_binary_data

# Inputs, targets
X, T = get_binary_data() 

# Random initialization
D = X.shape[1] #nb of input features
W = np.random.randn(D) # Dx1 in this case
b = 0

def sigmoid(z):
    return 1/(1+np.exp(-z))

def feedforward(X, W, b):
    return sigmoid(X.dot(W) + b)

P_T_given_X = feedforward(X, W, b)
predictions = np.round(P_T_given_X)

def classification_rate2(P, T):
    # P == T: booloean matrix size P 
    return np.mean(P == T)

print('Score accuracy: {}'.format(classification_rate2(predictions, T)))

