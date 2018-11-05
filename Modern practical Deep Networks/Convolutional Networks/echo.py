import wave
import sys 
import numpy as np
import matplotlib.pyplot as plt

from scipy.io.wavfile import write

# On Ubuntu 16.04 if you right-click on the audo file > properties > audio, you can see:
# sampling rate = 44100 Hz (FREQUENCE DU SON)
# bits per sample = 32

# The first is quantization in TIME (frequency)
# The second is quantization in AMPLITUDE

# 16 bits to represent the amplitude : 2^32 = 4294967296 is how many different sound levels we have (in informatics)
# For images, it will be 3 octets: 2^8 * 2^8 * 2^8 = 2^24 is how many different colors we can represent

# Original sample: https://www.youtube.com/watch?v=vCWJoGSmrM8

spf = wave.open('mario.wav', 'r')

# Extract Raw Audio from Wav File and convert it into a numpy array
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int32')
print("numpy signal:", signal.shape)

plt.plot(signal)
plt.title("Mario without echo")
plt.savefig('mario_no_echo.png')

delta = np.array([1.])
noecho = np.convolve(signal, delta)
print("noecho signal:", noecho.shape)
assert(np.abs(noecho[:len(signal)] - signal).sum() < 0.000001)

noecho = noecho.astype(np.int32) # make sure you do this, otherwise, you will get VERY LOUD NOISE
write('noecho.wav', 44100, noecho)

# Filter for one second
filt = np.zeros(44100)
filt[0] = 1
filt[11025] = 0.8
filt[22050] = 0.6
filt[33075] = 0.4
filt[44099] = 0.2

echo = np.convolve(signal, filt)

echo = echo.astype(np.int32) # make sure you do this, otherwise, you will get VERY LOUD NOISE
write('echo.wav', 44100, echo)

plt.plot(echo)
plt.title("Mario with echo")
plt.savefig('mario_echo.png')


