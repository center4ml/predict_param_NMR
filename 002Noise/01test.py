import math
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

#Demonstrates that spectra generated in tensorflow are identical to those generated in numpy. Randomly takes 1 to 9 peaks and distributes them at random regardless of the YOLO grid. Displays spectra generated in tensorflow and numpy. Calculates mean squared differences between them.

#Ranges of speed, frequency, damping coefficient and amplitude in units used in experiment
sp_min, sp_max = -10., 10.
fr_min, fr_max = 0., 256.
dc_min, dc_max = 0.1, 10.
am_min, am_max = 0.1, 5.

#Range of noise dispersion in units used in experiment
nd_min, nd_max = 0.1, 5.

#Slightly modified constants from Daniel
series = 20
dwmin, dwmax = -10., 10.

#Number of pixels along the speed and frequency dimension
sp_pixels, fr_pixels = 2000, 256

#Random number of peaks from 1 to 9
peaks = tf.random.uniform([], 1, 10, dtype = tf.int32)

#Random peak parameters in units used in experiment
sp_values = tf.random.uniform([peaks], sp_min, sp_max)
fr_values = tf.random.uniform([peaks], fr_min, fr_max)
dc_values = tf.random.uniform([peaks], dc_min, dc_max)
am_values = tf.random.uniform([peaks], am_min, am_max)

#Random noise dispersion in units used in experiment
nd_value = tf.random.uniform([], nd_min, nd_max)

#Gaussian noise
noise = tf.complex(tf.random.normal([2 * fr_pixels, series], 0., nd_value),
                   tf.random.normal([2 * fr_pixels, series], 0., nd_value)) #(512, 20)

#Auxiliary arrays for generating spectrum in tensorflow
t = tf.complex(tf.range(2 * fr_pixels, dtype = tf.float32) / fr_pixels / 2., 0.) #(512)
a = tf.range(series, dtype = tf.float32) #(20)
b = 2. * math.pi * \
    tf.complex(0., tf.range(dwmin, dwmax, (dwmax - dwmin) / sp_pixels, dtype = tf.float32))[:, None] * t #(2000, 512)

#Spectrum generated in tensorflow
f = a[:, None] * sp_values + fr_values + fr_pixels / 2. #(20)
fid = tf.complex(am_values, 0.) * tf.exp(2. * math.pi * t[:, None, None] * (tf.complex(-dc_values, f))) * \
      tf.cast(tf.logical_and(0.8 * fr_pixels / 2. <= f, f <= 2. * fr_pixels - 0.8 * fr_pixels / 2.), tf.complex64) #(512, 20, 3)
fid = tf.reduce_sum(fid, 2) #(512, 20)
fid = fid + noise
fid = tf.concat([fid[: 1] / 2., fid[1:]], 0)
p = tf.exp(-b[:, :, None] * tf.complex(a, 0.)) * fid #(2000, 512, 20)
spectrum_tf = tf.math.real(tf.signal.fft(tf.reduce_sum(p, 2)))[:, fr_pixels // 2 : 2 * fr_pixels - fr_pixels // 2] #(2000, 256)

#Plot the tensorflow spectrum
plt.figure()
plt.imshow(tf.transpose(spectrum_tf))

#Peak parameters in units used in experiment
sp_values = np.array(sp_values, np.float64)
fr_values = np.array(fr_values, np.float64)
dc_values = np.array(dc_values, np.float64)
am_values = np.array(am_values, np.float64)

#Gaussian noise
noise = np.array(noise, np.complex128)

#Auxiliary arrays for generating spectrum in numpy
t = np.arange(2 * fr_pixels) / fr_pixels / 2 #(512)
a = np.arange(series) #(20)
b = 2 * np.pi * 1j * np.arange(dwmin, dwmax, (dwmax - dwmin) / sp_pixels)[:, None] * t #(2000, 512)
 
#Spectrum generated in numpy
f = a[:, None] * sp_values + fr_values + fr_pixels / 2 #(20, 3)
fid = am_values * np.exp(2 * np.pi * t[:, None, None] * (1j * f - dc_values)) * \
      np.logical_and(0.8 * fr_pixels / 2 <= f, f <= 2 * fr_pixels - 0.8 * fr_pixels / 2) #(512, 20, 3)
fid = fid.sum(2) #(512, 20)
fid += noise #(512, 20)
fid[0] /= 2
p = np.exp(-b[:, :, None] * a) * fid #(2000, 512, 20)
spectrum_np = np.fft.fft(p.sum(2))[:, fr_pixels // 2 : 2 * fr_pixels - fr_pixels // 2].real #(2000, 256)

#Plot the numpy spectrum
plt.figure()
plt.imshow(spectrum_np.T)

#Compare
print(np.square(spectrum_tf - spectrum_np).mean())

plt.show()
