import math
import numpy as np
import tensorflow as tf

#Evaluates the model performance on a large test batch. Calculates objectness accuracy. Calculates root-mean-square errors, mean errors and error standard deviations of the predicted peak parameters.

#Ranges of speed, frequency, damping coefficient and amplitude in units used in experiment
sp_min, sp_max = -10., 10.
fr_min, fr_max = 0., 256.
dc_min, dc_max = 0.1, 10.
am_min, am_max = 0.1, 5.

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

#Load the model weights
model.load_weights('05extract.h5')

#Auxiliary arrays for generating spectrum in tensorflow
t = tf.complex(tf.range(2 * fr_pixels, dtype = tf.float32) / fr_pixels / 2., 0.)
a = tf.range(series, dtype = tf.float32)
b = 2. * math.pi * \
    tf.complex(0., tf.range(dwmin, dwmax, (dwmax - dwmin) / sp_pixels, dtype = tf.float32))[:, None] * t

#Takes peak parameters and returns the resulting spectrum
@tf.function
def get(ob_labels, sp_values, fr_values, dc_values, am_values):
    f = a[:, None, None] * sp_values + fr_values + fr_pixels / 2.
    fid = tf.complex(ob_labels * am_values, 0.) * \
          tf.exp(2. * math.pi * t[:, None, None, None] * (tf.complex(-dc_values, f))) * \
          tf.cast(tf.logical_and(0.8 * fr_pixels / 2. <= f, f <= 2. * fr_pixels - 0.8 * fr_pixels / 2.), tf.complex64)
    fid = tf.reduce_sum(fid, [2, 3])
    fid = tf.concat([fid[: 1] / 2., fid[1:]], 0)
    p = tf.exp(-b[:, :, None] * tf.complex(a, 0.)) * fid
    return tf.math.real(tf.signal.fft(tf.reduce_sum(p, 2)))[:, fr_pixels // 2 : 2 * fr_pixels - fr_pixels // 2]

#Draws random peak parameters and yields the resulting spectrum
def generate():
    while True:
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
        spectrum = get(ob_labels, sp_values, fr_values, dc_values, am_values)
        yield spectrum, (ob_labels, sp_shifts, fr_shifts, dc_shifts, am_shifts)

#Normalizes spectra to zero mean and unit variance
@tf.function
def normalize(spectrum, target):
    spectrum = (spectrum - 180.45589) / 217.40448
    return spectrum, target

#Batched dataset of normalized spectra from the above generator
dataset = tf.data.Dataset.from_generator(generate,
                                         output_signature = (tf.TensorSpec([sp_pixels, fr_pixels]),
                                                             (tf.TensorSpec([sp_cells, fr_cells]),
                                                              tf.TensorSpec([sp_cells, fr_cells]),
                                                              tf.TensorSpec([sp_cells, fr_cells]),
                                                              tf.TensorSpec([sp_cells, fr_cells]),
                                                              tf.TensorSpec([sp_cells, fr_cells]))))
dataset = dataset.map(normalize).batch(16384)

#Generated spectra and true objectness labels and peak shifts
spectrum, (ob_labels0, sp_shifts0, fr_shifts0, dc_shifts0, am_shifts0) = next(iter(dataset))

#Predicted objectness logits and peak shifts
ob_logits1, sp_shifts1, fr_shifts1, dc_shifts1, am_shifts1 = model.predict(spectrum)

#True peak shifts
sp_shifts0 = sp_shifts0.numpy()
fr_shifts0 = fr_shifts0.numpy()
dc_shifts0 = dc_shifts0.numpy()
am_shifts0 = am_shifts0.numpy()

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

#True objectness labels
ob_labels0 = ob_labels0.numpy()
ob_labels0 = ob_labels0 > 0.5

#Predicted objectness labels
ob_labels1 = ob_logits1 > 0.

#Objectness accuracy
ob_labels_acc = np.mean(ob_labels1 == ob_labels0)

#Print objectness accuracy
print('ob_labels_acc:', ob_labels_acc)
print()

#Objectness labels for correctly predicted peaks
ob_labels = np.logical_and(ob_labels1, ob_labels0)

#True shifts for correctly predicted peaks
sp_shifts0 = sp_shifts0[ob_labels]
fr_shifts0 = fr_shifts0[ob_labels]
dc_shifts0 = dc_shifts0[ob_labels]
am_shifts0 = am_shifts0[ob_labels]

#Predicted shifts for correctly predicted peaks
sp_shifts1 = sp_shifts1[ob_labels]
fr_shifts1 = fr_shifts1[ob_labels]
dc_shifts1 = dc_shifts1[ob_labels]
am_shifts1 = am_shifts1[ob_labels]

#Root-mean-squared errors of shifts for correctly predicted peaks
sp_shifts_rmse = np.sqrt(np.square(sp_shifts1 - sp_shifts0).mean())
fr_shifts_rmse = np.sqrt(np.square(fr_shifts1 - fr_shifts0).mean())
dc_shifts_rmse = np.sqrt(np.square(dc_shifts1 - dc_shifts0).mean())
am_shifts_rmse = np.sqrt(np.square(am_shifts1 - am_shifts0).mean())

#Print root-mean-squared errors of shifts for correctly predicted peaks
print('sp_shifts_rmse:', sp_shifts_rmse)
print('fr_shifts_rmse:', fr_shifts_rmse)
print('dc_shifts_rmse:', dc_shifts_rmse)
print('am_shifts_rmse:', am_shifts_rmse)
print()

#Mean errors of shifts for correctly predicted peaks
sp_shifts_me = np.mean(sp_shifts1 - sp_shifts0)
fr_shifts_me = np.mean(fr_shifts1 - fr_shifts0)
dc_shifts_me = np.mean(dc_shifts1 - dc_shifts0)
am_shifts_me = np.mean(am_shifts1 - am_shifts0)

#Print mean errors of shifts for correctly predicted peaks
print('sp_shifts_me:', sp_shifts_me)
print('fr_shifts_me:', fr_shifts_me)
print('dc_shifts_me:', dc_shifts_me)
print('am_shifts_me:', am_shifts_me)
print()

#Standard deviations of errors in shifts for correctly predicted peaks
sp_shifts_estd = np.sqrt(np.square(sp_shifts1 - sp_shifts0 - sp_shifts_me).mean())
fr_shifts_estd = np.sqrt(np.square(fr_shifts1 - fr_shifts0 - fr_shifts_me).mean())
dc_shifts_estd = np.sqrt(np.square(dc_shifts1 - dc_shifts0 - dc_shifts_me).mean())
am_shifts_estd = np.sqrt(np.square(am_shifts1 - am_shifts0 - am_shifts_me).mean())

#Print standard deviations of errors in shifts for correctly predicted peaks
print('sp_shifts_estd:', sp_shifts_estd)
print('fr_shifts_estd:', fr_shifts_estd)
print('dc_shifts_estd:', dc_shifts_estd)
print('am_shifts_estd:', am_shifts_estd)
print()

#True units for correctly predicted peaks
sp_units0 = sp_units0[ob_labels]
fr_units0 = fr_units0[ob_labels]
dc_units0 = dc_units0[ob_labels]
am_units0 = am_units0[ob_labels]

#Predicted units for correctly predicted peaks
sp_units1 = sp_units1[ob_labels]
fr_units1 = fr_units1[ob_labels]
dc_units1 = dc_units1[ob_labels]
am_units1 = am_units1[ob_labels]

#Root-mean-squared errors of units for correctly predicted peaks
sp_units_rmse = np.sqrt(np.square(sp_units1 - sp_units0).mean())
fr_units_rmse = np.sqrt(np.square(fr_units1 - fr_units0).mean())
dc_units_rmse = np.sqrt(np.square(dc_units1 - dc_units0).mean())
am_units_rmse = np.sqrt(np.square(am_units1 - am_units0).mean())

#Print root-mean-squared errors of units for correctly predicted peaks
print('sp_units_rmse:', sp_units_rmse)
print('fr_units_rmse:', fr_units_rmse)
print('dc_units_rmse:', dc_units_rmse)
print('am_units_rmse:', am_units_rmse)
print()

#Mean errors of units for correctly predicted peaks
sp_units_me = np.mean(sp_units1 - sp_units0)
fr_units_me = np.mean(fr_units1 - fr_units0)
dc_units_me = np.mean(dc_units1 - dc_units0)
am_units_me = np.mean(am_units1 - am_units0)

#Print mean errors of units for correctly predicted peaks
print('sp_units_me:', sp_units_me)
print('fr_units_me:', fr_units_me)
print('dc_units_me:', dc_units_me)
print('am_units_me:', am_units_me)
print()

#Standard deviations of errors in units for correctly predicted peaks
sp_units_estd = np.sqrt(np.square(sp_units1 - sp_units0 - sp_units_me).mean())
fr_units_estd = np.sqrt(np.square(fr_units1 - fr_units0 - fr_units_me).mean())
dc_units_estd = np.sqrt(np.square(dc_units1 - dc_units0 - dc_units_me).mean())
am_units_estd = np.sqrt(np.square(am_units1 - am_units0 - am_units_me).mean())

#Print standard deviations of errors in units for correctly predicted peaks
print('sp_units_estd:', sp_units_estd)
print('fr_units_estd:', fr_units_estd)
print('dc_units_estd:', dc_units_estd)
print('am_units_estd:', am_units_estd)
print()
