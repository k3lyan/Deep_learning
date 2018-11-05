import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1+np.exp(-z))

def cross_entropy(T,Y):
    J = 0
    for i in range(N):
        if (T[i]==1):
            J -= np.log(Y[i])
        else:
            J -= np.log(1 - Y[i])
    return J

N = 1000
D = 2

R_inner = 5
R_outer = 10

print(type(N//2))
print(type(N/2))

R1 = np.random.randn(N//2) + R_inner
# np.random.random: returns random float in [0.0, 1.0[
theta = 2*np.pi*np.random.random(N//2)
X_inner = np.concatenate([[R1*np.cos(theta)], [R1*np.sin(theta)]]).T
print(X_inner.shape)
R2 = np.random.randn(N//2) + R_outer
theta = 2*np.pi*np.random.random(N//2)
X_outer = np.concatenate([[R2*np.cos(theta)], [R2*np.sin(theta)]]).T

X = np.concatenate([X_inner, X_outer])
T = np.array([0]*(N//2)+[1]*(N//2))
print(X)
print(X.shape)
plt.scatter(X[:,0], X[:,1], c=T)
plt.savefig('donut.png')

ones = np.array([[1]*N]).T
# print(ones)
r = np.zeros((N,1))
for i in range(N):
    r[i] = np.sqrt(X[i,:].dot(X[i,:]))

Xb = np.concatenate((ones,r,X), axis = 1)
W = np.random.rand(D+2)
Z = Xb.dot(W)
Y = sigmoid(Z)
learning_rate = 0.0001
error = []
for i in range(5000):
    e = cross_entropy(T,Y)
    error.append(e)
    '''
    if (i % 100 == 0):
        print(e)
    '''
    #GD
    W += learning_rate * (Xb.T.dot(T-Y) - 0.01* W)
    Y = sigmoid(Xb.dot(W))

plt.plot(error)
plt.title('cross_entropy')
plt.savefig('cross_entropy.png')
print('Final W: {}'.format(W))
print('Classification rate: {}'.format(1 - np.abs(T - np.round(Y)).sum()/N))


