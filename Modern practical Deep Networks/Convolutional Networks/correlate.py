import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

X = np.random.randn(1000)
C = correlate(X, X)
plt.plot(C)
plt.show()

Y = np.empty(1000)
Y[:900] = X[100:]
Y[900:] = X[:100]
C2 = correlate(X,Y)
plt.plot(C2)
plt.show()
