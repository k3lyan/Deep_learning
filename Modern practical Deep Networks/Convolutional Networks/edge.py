import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.signal import convolve2d

# Load Lena
img = mpimg.imread('lena.png')
# Edge detection work only on gray-scale pic
bw = img.mean(axis=2)

# Sobel operator: approximate horizontal gradient
Hx = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
    ], dtype = np.float32)

# Sobel operator: approximate vertical gradient
Hy = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
    ], dtype = np.float32)

Gx = convolve2d(bw, Hx)
plt.imshow(Gx, cmap='gray')
plt.show()

Gy = convolve2d(bw, Hy)
plt.imshow(Gy, cmap='gray')
plt.show()

# Approximate norm gradient
G = np.sqrt(Gx*Gx + Gy*Gy)
plt.imshow(G, cmap='gray')
plt.show()

# Gradient direction
theta = np.arctan2(Gy, Gx)
plt.imshow(theta, cmap='gray')
plt.show()
