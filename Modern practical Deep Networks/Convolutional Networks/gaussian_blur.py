import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load Lena
img = mpimg.imread('lena.png')

plt.imshow(img)
plt.show()

# Make it B&W since the 2D Convolution is only define for 2D matrices
# So we take the average on the second axis
bw = img.mean(axis=2)
plt.imshow(bw, cmap='gray')
plt.show()

# Let's create a Gaussian filter
# It's gonna be a 20X20 filter
# can be other sizes too
# g(x) ~= exp(-(x-mu)^2/(2*sigma^2))
# Here we divide by a large nb
D = 20
W = np.zeros((D, D))
for i in range(D):
    for j in range(D):
        dist = (i-9.5)**2 + (j-9.5)**2
        W[i, j] = np.exp(-dist/50.)
# Normalize the filter to get value between 0 and 1
W /= W.sum()

# Let's see what the filter looks like
plt.imshow(W, cmap='gray')
plt.show()

# Now, the convolution between the B&W pic and our filter W
gausse2d = convolve2d(bw,W)
plt.imshow(gausse2d, cmap='gray')
plt.show()

# Why do we see this black layer all around ?
print('bw.shape: {}'.format(bw.shape))
print('Gaussian filter shape: {}'.format(W.shape))
print('gausse2d.shape: {}'.format(gausse2d.shape))
# After convolution, the output signal is of size: N1 + N2 -1

# If we want to make it as the same size of the input
gausse2d_same_size = convolve2d(bw, W, mode='same')
plt.imshow(gausse2d_same_size, cmap='gray')
plt.show()

# With colors
colored = np.zeros(img.shape)
print('initial image shape: {}'.format(img.shape))
for i in range(3):
    colored[:,:,i] = convolve2d(img[:,:,i], W, mode='same')

plt.imshow(colored)
plt.show()
