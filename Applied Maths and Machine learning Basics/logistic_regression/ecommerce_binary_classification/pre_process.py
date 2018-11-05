import numpy as np
import pandas as pd

def get_data():
    df = pd.read_csv('ecommerce_data.csv')
    # Get your data into a numpy array
    data = df.as_matrix()
    # Inputs
    X = data[:,:-1]
    # Output to compare with: TARGETS
    T = data[:,-1]
    
    # Numerical values normalized, otherwise non-sense for activation functions
    # n_products_viewed (int) col1 
    X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
    # visit_duration (float) col2 
    X[:,2] = (X[:,2] - X[:,2].mean()) / X[:,2].std() 
    
    # Categorical values: one hot-encoding
    # time_of_day (0/1/2/3)
    N, D = X.shape
    X2 = np.zeros((N,D+3))
    X2[:,0:(D-1)] = X[:,0:(D-1)]
    # One-hot encoding: first method
    for n in range(N):
        t = int(X[n,D-1])
        X2[n,D-1+t] = 1
    # One-hot encoding: second method
    OHE = np.zeros((N, 4))
    OHE[np.arange(N), X[:,D-1].astype(np.int32)] = 1
    # we then assign: X2[:,-4:] = OHE
    # Verify if the 2 methods work
    assert(np.abs(X2[:,-4:] - OHE).sum() < 10e-10)
    return X2, T

# For the logistic class we only predict binary data bounce (0) and add_to_cart (1)
# So we have to get rid of the 2 other types of action: begin_checkout(2), finish_checkout(3)
def get_binary_data():
    X, T = get_data()
    X2 = X[T <= 1]
    T2 = T[T <= 1]
    return X2, T2

get_binary_data()
