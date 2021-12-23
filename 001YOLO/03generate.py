import math
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

#Demonstrates that spectra generated in tensorflow and numpy are identical to those generated in the original python code from Daniel. Randomly takes 1 to 9 peaks and distributes them at random so that each YOLO cell contains at most one peak.

#Slightly modified constants from Daniel
series = 20
dwmin, dwmax = -10., 10.

#Ranges of speed, frequency, damping coefficient and amplitude in units used in experiment
sp_min, sp_max = -10., 10.
fr_min, fr_max = 0., 256.
dc_min, dc_max = 0.1, 10.
am_min, am_max = 0.1, 5.

#Number of pixels and cells along the speed and frequency dimension
sp_pixels, fr_pixels = 256, 256
sp_cells, fr_cells = 16, 16

#Offsets of cells relative to the entire grid in range from 0 to number of cells
fr_offsets, sp_offsets = tf.meshgrid(tf.range(fr_cells, dtype = tf.float32),
                                     tf.range(sp_cells, dtype = tf.float32))

#Random number of peaks from 1 to 9
count = tf.random.uniform([], 1, 10, dtype = tf.int32)

#Random objectness labels as 1. or 0. if a cell contains a peak or not
ob_labels = tf.concat([tf.ones([count]), tf.zeros([sp_cells * fr_cells - count])], -1)
ob_labels = tf.random.shuffle(ob_labels)
ob_labels = tf.reshape(ob_labels, [sp_cells, fr_cells])

#Random peak parameters relative to each cell in range from 0 to 1
sp_shifts = tf.random.uniform([sp_cells, fr_cells])
fr_shifts = tf.random.uniform([sp_cells, fr_cells])
dc_shifts = tf.random.uniform([sp_cells, fr_cells])
am_shifts = tf.random.uniform([sp_cells, fr_cells])

#Peak parameters relative to the entire grid in range from 0 to 1
sp_units = (sp_offsets + sp_shifts) / sp_cells
fr_units = (fr_offsets + fr_shifts) / fr_cells
dc_units = dc_shifts
am_units = am_shifts

#Peak parameters in units used in experiment
sp_values = sp_units * (sp_max - sp_min) + sp_min
fr_values = fr_units * (fr_max - fr_min) + fr_min
dc_values = dc_units * (dc_max - dc_min) + dc_min
am_values = am_units * (am_max - am_min) + am_min

#Auxiliary arrays for generating spectrum in tensorflow
t = tf.complex(tf.range(2 * fr_pixels, dtype = tf.float32) / fr_pixels / 2., 0.)
a = tf.range(series, dtype = tf.float32)
b = 2. * math.pi * \
    tf.complex(0., tf.range(dwmin, dwmax, (dwmax - dwmin) / sp_pixels, dtype = tf.float32))[:, None] * t

#Spectrum generated in tensorflow
f = a[:, None, None] * sp_values + fr_values + fr_pixels / 2.
fid = tf.complex(ob_labels * am_values, 0.) * \
      tf.exp(2. * math.pi * t[:, None, None, None] * (tf.complex(-dc_values, f))) * \
      tf.cast(tf.logical_and(0.8 * fr_pixels / 2. <= f, f <= 2. * fr_pixels - 0.8 * fr_pixels / 2.), tf.complex64)
fid = tf.reduce_sum(fid, [2, 3])
fid = tf.concat([fid[: 1] / 2., fid[1:]], 0)
p = tf.exp(-b[:, :, None] * tf.complex(a, 0.)) * fid
spectrum_tf = tf.math.real(tf.signal.fft(tf.reduce_sum(p, 2)))[:, fr_pixels // 2 : 2 * fr_pixels - fr_pixels // 2]

#Plot the tensorflow spectrum
plt.figure()
plt.imshow(tf.transpose(spectrum_tf))

#Objectness labels
ob_labels = np.array(ob_labels, np.bool)

#Peak parameters in units used in experiment
sp_values = np.array(sp_values, np.float64)
fr_values = np.array(fr_values, np.float64)
dc_values = np.array(dc_values, np.float64)
am_values = np.array(am_values, np.float64)

#Auxiliary arrays for generating spectrum in numpy
t = np.arange(2 * fr_pixels) / fr_pixels / 2
a = np.arange(series)
b = 2 * np.pi * 1j * np.arange(dwmin, dwmax, (dwmax - dwmin) / sp_pixels)[:, None] * t

#Spectrum generated in numpy
f = a[:, None, None] * sp_values + fr_values + fr_pixels / 2
fid = ob_labels * am_values * \
      np.exp(2 * np.pi * t[:, None, None, None] * (1j * f - dc_values)) * \
      np.logical_and(0.8 * fr_pixels / 2 <= f, f <= 2 * fr_pixels - 0.8 * fr_pixels / 2)
fid = fid.sum((2, 3))
fid[0] /= 2
p = np.exp(-b[:, :, None] * a) * fid
spectrum_np = np.fft.fft(p.sum(2))[:, fr_pixels // 2 : 2 * fr_pixels - fr_pixels // 2].real

#Plot the numpy spectrum
plt.figure()
plt.imshow(spectrum_np.T)

#Peak parameters in units used in experiment
sp_values = sp_values[ob_labels].flatten()
fr_values = fr_values[ob_labels].flatten()
dc_values = dc_values[ob_labels].flatten()
am_values = am_values[ob_labels].flatten()

#Spectrum generated in original python code
n2 = 2 * fr_pixels
freq_shift = int(fr_pixels / 2)
shifted_freq = [freq + freq_shift for freq in fr_values]
fid = np.zeros((series, n2), dtype = "complex128")
t = np.linspace(0, 1, n2, endpoint = False)
for i in range(series):
    for k in range(np.shape(am_values)[0]):
        tot_freq = shifted_freq[k] + i * sp_values[k]
        if (0.8 * freq_shift <= tot_freq <= n2 - 0.8 * freq_shift):
            exponent = (2 * np.pi * 1j * (tot_freq + 1j * dc_values[k]) * t)
            fid[i] = np.add(fid[i], am_values[k] * np.e ** exponent)
for i in range(series):
    fid[i][0] /= 2
dw_arr = np.linspace(dwmin, dwmax, sp_pixels, endpoint = False)
p = np.zeros((len(dw_arr), *np.shape(fid)), dtype = "complex128")
a = 2 * np.pi * 1j * np.linspace(0, 1, n2, endpoint = False)
for i in range(len(dw_arr)):
    b = a * dw_arr[i]
    for k in range(series):
        p[i][k] = fid[k] * np.e ** (-b * k)
pr = np.zeros((len(dw_arr), n2), dtype = "complex128")
for i in range(len(dw_arr)):
    for j in range(series):
        pr[i] += p[i][j]
spectrum_py = np.fft.fft(pr)[:, freq_shift : n2 - freq_shift].real

#Plot the python spectrum
plt.figure()
plt.imshow(spectrum_py.T)

#Compare
print(np.square(spectrum_tf - spectrum_py).mean())
print(np.square(spectrum_np - spectrum_py).mean())

plt.show()
