import numpy as np
import matplotlib.pyplot as plt

def NN_forward(X, W0, b0, W1, b1):
    # A1 --> (N, M)
    A1 = 1 / (1 + np.exp(-(X.dot(W0) + b0)))
    # U --> (N, K)
    U = A1.dot(W1) - b1
    expU = np.exp(U)
    # Y --> (N, K)
    Y = expU / expU.sum(axis=1, keepdims=True)
    return A1, Y

# Classification_rate : num_correct / num_total
def classification_rate(P, T):
    n_correct = 0
    n_total = 0
    for i in range(len(P)):
        n_total += 1
        if (P[i] == T[i]):
            n_correct += 1
    return float(n_correct) / n_total

def derivative_W1(A1, T, Y):
    N, K = T.shape
    M = A1.shape[1]
    
    # slow version
    #der1 = np.zeros((M, K))
    #for n in range(N):
    #    for m in range(M):
    #        for k in range(K):
    #            der1[m, k] += (T[n, k] - Y[n,k]) * A1[n, m]
    
    #der2 = np.zeros((M, K))
    #for n in range(N):
    #    for k in range(K):
    #        der2[:, k] += (T[n, k] - Y[n,k]) * A1[n, :]
    #assert(np.abs(der1-der2).sum() < 10e-10)
    
    #der3 = np.zeros((M, K))
    #for n in range(N):
    #    der3 += np.outer(A1[n]*(T[n] - Y[n]))
    #assert(np.abs(der2-der3).sum() < 10e-10)
    
    der4 = A1.T.dot(T - Y)
    #assert(np.abs(der3-der4).sum() < 10e-10)

    return der4

def derivative_b1(T, Y):
    return (T - Y).sum(axis = 0)

def derivative_W0(X, A1, T, Y, W1):
    #N, D = X.shape
    #M, K = W1.shape
    
    # slow version
    #der1 = np.zeros((D, M))
    #for n in range(N):
    #    for k in range(K):
    #        for m in range(M):
    #            for d in range(D):
    #                der1[d, m] =+ (T[n, k] - Y[n, k]) * W1[m, k] * A1[n, m] * (1 - A1[n, m]) * X[n, d]
    
    dA1 = (T-Y).dot(W1.T) * A1 * (1-A1)
    return X.T.dot(dA1)

def derivative_b0(A1, T, Y, W1):
    return ((T - Y).dot(W1.T) *A1 * (1 - A1)).sum(axis = 0)

def cost(T,Y):
    tot = T * np.log(Y)
    return tot.sum()

def main():
    # Create the data
    ## Nb of samples by class: we work with 3 populations here so N = 3*N_class
    N_class = 500
    ## nb of input features
    D = 2    
    ## hidden layer size
    M = 3    
    ## nb of classes
    K = 3    

    X1 = np.random.randn(N_class, D) + np.array([0, -2])
    X2 = np.random.randn(N_class, D) + np.array([2, -2])
    X3 = np.random.randn(N_class, D) + np.array([-2, -2])
    X = np.vstack([X1, X2, X3])
    
    t = np.array(N_class*[0] + N_class*[1] + N_class*[2])
    #print(len(t))
    N = len(t)
    
    # One-hot encoding target marix
    T = np.zeros((N, K))
    for i in range(N):
        T[i, t[i]] = 1
    
    # Have a look at the inputs:
    plt.scatter(X[:,0], X[:,1], c=t, s=100, alpha=0.5)
    plt.savefig('inputs.png')

    # Randomly initialized the weights
    W0 = np.random.randn(D,M)
    b0 = np.random.randn(M)
    W1 = np.random.randn(M,K)
    b1 = np.random.randn(K)

    learning_rate = 10e-7
    costs = []
    iterations = 100000
    for epoch in range(iterations):
        hidden, output = NN_forward(X, W0, b0, W1, b1)
        if (epoch % 100 == 0):
            c = cost(T, output)
            P = np.argmax(output, axis=1)
            r = classification_rate(t, P)
            print('cost: {}\t classification_rate: {}'.format(c, r))
            costs.append(c)
        # Gradient-ascent
        W1 += learning_rate * derivative_W1(hidden, T, output)
        b1 += learning_rate * derivative_b1(T, output)
        W0 += learning_rate * derivative_W0(X, hidden, T, output, W1)
        b0 += learning_rate * derivative_b0(hidden, T, output, W1)

    plt.plot(costs)
    plt.savefig('Cost_evolution_100000_epochs.png')


if __name__ == '__main__':
    main()
         
