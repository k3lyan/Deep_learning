import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime

# Lets' build and compare our own convolution function for 2D images
# if the original image is of size N1*N2, the filter of size M1*M2
# then the output image of size (N1+M1-1)*(N2+M2-1)

##very slow version (4 loops)
# Elapsed time 4 loops: 0:01:47.919295
#def convolve2d(X, W):
#    t0 = datetime.now()
#    n1, n2 = X.shape
#    m1, m2 = W.shape
#    Y = np.zeros((n1+m1-1, n2+m2-1))
#    for i in range(n1+m1-1):
#        for ii in range(m1):
#            for j in range(n2+m2-1):
#                for jj in range(m2):
#                    if i>=ii and j>=jj and i-ii<n1 and j-jj<n2:
#                        Y[i,j] += W[ii, jj] * X[i-ii, j-jj]
#    print('Elapsed time 4 loops: {}'.format(datetime.now() - t0))
#    return Y

##slow version (2 loops)
# Notice that 2 of these loops really only multiply X[i,j] 
# by every part of the filter W
# Elapsed time 2 loops: 0:00:03.426583
#def convolve2d(X, W):
#    t0 = datetime.now()
#    n1, n2 = X.shape
#    m1, m2 = W.shape
#    Y = np.zeros((n1+m1-1, n2+m2-1))
#    for i in range(n1):
#        for j in range(n2):
#            Y[i:i+m1, j:j+m2] += X[i, j]*W
#    print('Elapsed time 2 loops: {}'.format(datetime.now() - t0))
#    return Y

# Half-padding: 'size(output_image) = size(input_image)'
def convolve2d_same(X, W):
    t0 = datetime.now()
    n1, n2 = X.shape
    m1, m2 = W.shape
    Y = np.zeros((n1+m1-1, n2+m2-1))
    for i in range(n1):
        for j in range(n2):
            Y[i:i+m1, j:j+m2] = X[i,j]*W
    reduced = Y[m1//2:-m1//2+1,m2//2:-m2//2+1]
    print('Elapsed time 2 loops same sized: {}'.format(datetime.now() - t0))
    return reduced

# Full-padding: 'size(output_image) = N+M-1'
def convolve2d_small(X, W):
    t0 = datetime.now()
    n1, n2 = X.shape
    m1, m2 = W.shape
    Y = np.zeros((n1+m1-1, n2+m2-1))
    for i in range(n1):
        for j in range(n2):
            Y[i:i+m1, j:j+m2] = X[i,j]*W
    reduced = Y[m1:-m1+1,m2:-m2+1]
    print('Elapsed time 2 loops smaller sized: {}'.format(datetime.now() - t0))
    return reduced

# Load Lena
img = mpimg.imread('lena.png')
# make it B&W
bw = img.mean(axis=2)
plt.imshow(bw, cmap='gray')
plt.show()

# Create a Gaussian filter
M = 20 
W = np.zeros((M,M))
for i in range(M):
    for j in range(M):
        dist = np.sqrt((i-9.5)**2+(j-9.5)**2)
        W[i,j] = np.exp(-dist / 50.0)
#plt.imshow(W, cmap='gray')
#plt.show()

gausse2d = convolve2d_same(bw, W)
plt.imshow(gausse2d, cmap='gray')
plt.show()

gausse2d2 = convolve2d_small(bw, W)
plt.imshow(gausse2d2, cmap='gray')
plt.show()


