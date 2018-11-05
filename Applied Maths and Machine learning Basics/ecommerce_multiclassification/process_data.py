import pandas as pd
import numpy as np

def get_data():
    # get the data as a df
    df = pd.read_csv('ecommerce_data.csv')
    # transform the df into a numpy array
    data = df.as_matrix()

    X = data[:,:-1]
    T = data[:,-1]

    # normalize the continuous values
    X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
    X[:,2] = (X[:,2] - X[:,2].mean()) / X[:,2].std()

    N, D = X.shape

    # one_hot encoding for category features
    X2 = np.zeros((N, D+3))
    X2[:,0:(D-1)] = X[:,0:(D-1)]
    
    '''
    for i in range(N):
        t = int(X[i, D-1])
        X2[i,D-1 + t] = 1 
    '''

    OH = np.zeros((N,4))
    OH[np.arange(N), X[:,D-1].astype(np.int32)] = 1
    X2[:, -4:] = OH

    return X2, T

