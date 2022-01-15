import math
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

#Plots a random spectrum with generated and predicted peaks

#Ranges of speed, frequency, damping coefficient and amplitude in units used in experiment
sp_min, sp_max = -10., 10.
fr_min, fr_max = 0., 256.
dc_min, dc_max = 0.1, 10.
am_min, am_max = 0.1, 5.

#Range of noise dispersion in units used in experiment
nd_min, nd_max = 0., 0.

#Slightly modified constants from Daniel
series = 20
dwmin, dwmax = -10., 10.

#Number of pixels and cells along the speed and frequency dimension
sp_pixels, fr_pixels = 256, 256
sp_cells, fr_cells = 16, 16

#Offsets of cells relative to the entire grid in range from 0 to the number of cells
fr_offsets, sp_offsets = tf.meshgrid(tf.range(fr_cells, dtype = tf.float32),
                                     tf.range(sp_cells, dtype = tf.float32))

#Takes a spectrum and returns objectness and peak shifts on the grid
spectrum = tf.keras.Input([256, 256])
signal = tf.keras.layers.Reshape([256, 256, 1])(spectrum)
signal = tf.keras.layers.Conv2D(32, 3, 1, 'same', activation = 'relu')(signal)
signal = tf.keras.layers.Conv2D(32, 3, 1, 'same', activation = 'relu')(signal)
signal = tf.keras.layers.MaxPool2D()(signal) #(128, 128, 32)
signal = tf.keras.layers.Conv2D(64, 3, 1, 'same', activation = 'relu')(signal)
signal = tf.keras.layers.Conv2D(64, 3, 1, 'same', activation = 'relu')(signal)
signal = tf.keras.layers.MaxPool2D()(signal) #(64, 64, 64)
signal = tf.keras.layers.Conv2D(128, 3, 1, 'same', activation = 'relu')(signal)
signal = tf.keras.layers.Conv2D(128, 3, 1, 'same', activation = 'relu')(signal)
signal = tf.keras.layers.MaxPool2D()(signal) #(32, 32, 128)
signal = tf.keras.layers.Conv2D(256, 3, 1, 'same', activation = 'relu')(signal)
signal = tf.keras.layers.Conv2D(256, 3, 1, 'same', activation = 'relu')(signal)
signal = tf.keras.layers.MaxPool2D()(signal) #(16, 16, 256)
signal = tf.keras.layers.Conv2D(512, 3, 1, 'same', activation = 'relu')(signal)
ob_logits = tf.keras.layers.Conv2D(1, 3, 1, 'same')(signal)
ob_logits = tf.keras.layers.Reshape([16, 16])(ob_logits)
sp_shifts = tf.keras.layers.Conv2D(1, 3, 1, 'same')(signal)
sp_shifts = tf.keras.layers.Reshape([16, 16])(sp_shifts)
fr_shifts = tf.keras.layers.Conv2D(1, 3, 1, 'same')(signal)
fr_shifts = tf.keras.layers.Reshape([16, 16])(fr_shifts)
dc_shifts = tf.keras.layers.Conv2D(1, 3, 1, 'same')(signal)
dc_shifts = tf.keras.layers.Reshape([16, 16])(dc_shifts)
am_shifts = tf.keras.layers.Conv2D(1, 3, 1, 'same')(signal)
am_shifts = tf.keras.layers.Reshape([16, 16])(am_shifts)
model = tf.keras.models.Model(spectrum, (ob_logits, sp_shifts, fr_shifts, dc_shifts, am_shifts))

#Load model weights
model.load_weights('05extract00.h5')

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

#Random noise dispersion relative to each cell in range from 0 to 1
nd_shift = tf.random.uniform([])

#Peak parameters relative to the entire grid in range from 0 to 1
sp_units = (sp_offsets + sp_shifts) / sp_cells
fr_units = (fr_offsets + fr_shifts) / fr_cells
dc_units = dc_shifts
am_units = am_shifts

#Noise dispersion relative to the entire grid in range from 0 to 1
nd_unit = nd_shift

#Peak parameters in units used in experiment
sp_values = sp_units * (sp_max - sp_min) + sp_min
fr_values = fr_units * (fr_max - fr_min) + fr_min
dc_values = dc_units * (dc_max - dc_min) + dc_min
am_values = am_units * (am_max - am_min) + am_min

#Noise dispersion in units used in experiment
nd_value = nd_unit * (nd_max - nd_min) + nd_min

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
fid = fid + tf.complex(tf.random.normal([2 * fr_pixels, series], 0., nd_value),
                       tf.random.normal([2 * fr_pixels, series], 0., nd_value))
fid = tf.concat([fid[: 1] / 2., fid[1:]], 0)
p = tf.exp(-b[:, :, None] * tf.complex(a, 0.)) * fid
spectrum = tf.math.real(tf.signal.fft(tf.reduce_sum(p, 2)))[:, fr_pixels // 2 : 2 * fr_pixels - fr_pixels // 2]

#Objectness labels from generator
ob_labels0 = ob_labels.numpy()

#Peak shifts from generator
sp_shifts0 = sp_shifts.numpy()
fr_shifts0 = fr_shifts.numpy()
dc_shifts0 = dc_shifts.numpy()
am_shifts0 = am_shifts.numpy()

#Spectrum normalized to zero mean and unit variance as required by the model
spectrum = (spectrum - 179.24811) / 216.3278 #noise 0.0
#spectrum = (spectrum - 180.86696) / 217.4606 #noise 0.1
#spectrum = (spectrum - 179.58424) / 217.31755 #noise 0.2
#spectrum = (spectrum - 179.61589) / 222.4815 #noise 0.5
#spectrum = (spectrum - 181.52248) / 239.33989 #noise 1.0
#spectrum = (spectrum - 179.96593) / 296.3282 #noise 2.0
#spectrum = (spectrum - 179.81058) / 550.0122 #noise 5.0

#Predicting objectness and peak parameters by passing the spectrum to the model
ob_logits, sp_shifts, fr_shifts, dc_shifts, am_shifts = model(spectrum[None])

#Offsets of cells relative to the entire grid in range from 0 to the number of cells
sp_offsets = sp_offsets.numpy()
fr_offsets = fr_offsets.numpy()

#Predicted objectness logits and labels
ob_logits1 = ob_logits[0].numpy()
ob_labels1 = np.where(ob_logits1 < 0., 0., 1.)

#Predicted peak shifts
sp_shifts1 = sp_shifts[0].numpy()
fr_shifts1 = fr_shifts[0].numpy()
dc_shifts1 = dc_shifts[0].numpy()
am_shifts1 = am_shifts[0].numpy()

#True peak units
sp_units0 = (sp_offsets + sp_shifts0) / sp_cells
fr_units0 = (fr_offsets + fr_shifts0) / fr_cells
dc_units0 = dc_shifts0
am_units0 = am_shifts0

#Predicted peak units
sp_units1 = (sp_offsets + sp_shifts1) / sp_cells
fr_units1 = (fr_offsets + fr_shifts1) / fr_cells
dc_units1 = dc_shifts1
am_units1 = am_shifts1

#True peak coordinates in pixels
sp_cords0 = (sp_units0 * sp_pixels).astype(np.int32)
fr_cords0 = (fr_units0 * sp_pixels).astype(np.int32)

#Predicted peak coordinates in pixels
sp_cords1 = (sp_units1 * sp_pixels).astype(np.int32)
fr_cords1 = (fr_units1 * sp_pixels).astype(np.int32)

#Spectrum normalized to range from 0 to 1 for plotting
spectrum = spectrum.numpy()
spectrum = spectrum - spectrum.min()
spectrum = spectrum / spectrum.max()

#Spectrum converted from grayscale to RGB
spectrum = np.array([spectrum, spectrum, spectrum]).transpose((1, 2, 0))

#Plot the grid in blue
spectrum[16 * np.arange(16), :, 2] = 1.
spectrum[:, 16 * np.arange(16), 2] = 1.

#Plot true peaks in green crosses
for sp_cell in range(sp_cells):
    for fr_cell in range(fr_cells):
        if ob_labels0[sp_cell, fr_cell] > 0.5:
            spectrum[sp_cords0[sp_cell, fr_cell] + 0, fr_cords0[sp_cell, fr_cell] + 0, :] = [0., 1., 0.]
            spectrum[sp_cords0[sp_cell, fr_cell] - 1, fr_cords0[sp_cell, fr_cell] - 1, :] = [0., 1., 0.]
            spectrum[sp_cords0[sp_cell, fr_cell] - 1, fr_cords0[sp_cell, fr_cell] + 1, :] = [0., 1., 0.]
            spectrum[sp_cords0[sp_cell, fr_cell] + 1, fr_cords0[sp_cell, fr_cell] - 1, :] = [0., 1., 0.]
            spectrum[sp_cords0[sp_cell, fr_cell] + 1, fr_cords0[sp_cell, fr_cell] + 1, :] = [0., 1., 0.]

#Plot predicted peaks in red pluses
for sp_cell in range(sp_cells):
    for fr_cell in range(fr_cells):
        if ob_labels1[sp_cell, fr_cell] > 0.5:
            spectrum[sp_cords1[sp_cell, fr_cell] + 0, fr_cords1[sp_cell, fr_cell] + 0, :] = [1., 0., 0.]
            spectrum[sp_cords1[sp_cell, fr_cell] + 0, fr_cords1[sp_cell, fr_cell] - 1, :] = [1., 0., 0.]
            spectrum[sp_cords1[sp_cell, fr_cell] + 0, fr_cords1[sp_cell, fr_cell] + 1, :] = [1., 0., 0.]
            spectrum[sp_cords1[sp_cell, fr_cell] - 1, fr_cords1[sp_cell, fr_cell] + 0, :] = [1., 0., 0.]
            spectrum[sp_cords1[sp_cell, fr_cell] + 1, fr_cords1[sp_cell, fr_cell] + 0, :] = [1., 0., 0.]

#Display the spectrum with plotted grid and peaks
plt.imshow(spectrum)

plt.show()
