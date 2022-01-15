import math
import numpy as np
import pandas as pd
import sklearn.metrics 
import tensorflow as tf

#Evaluates the model performance on a large test batch. Calculates objectness accuracy. Calculates root-mean-square errors, mean errors and error standard deviations of the predicted peak parameters.

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

#Load the model weights
model.load_weights('05extract00.h5')

#Auxiliary arrays for generating spectrum in tensorflow
t = tf.complex(tf.range(2 * fr_pixels, dtype = tf.float32) / fr_pixels / 2., 0.)
a = tf.range(series, dtype = tf.float32)
b = 2. * math.pi * \
    tf.complex(0., tf.range(dwmin, dwmax, (dwmax - dwmin) / sp_pixels, dtype = tf.float32))[:, None] * t

#Takes peak parameters and returns the resulting spectrum
@tf.function
def get(ob_labels, sp_values, fr_values, dc_values, am_values, nd_value):
    f = a[:, None, None] * sp_values + fr_values + fr_pixels / 2.
    fid = tf.complex(ob_labels * am_values, 0.) * \
          tf.exp(2. * math.pi * t[:, None, None, None] * (tf.complex(-dc_values, f))) * \
          tf.cast(tf.logical_and(0.8 * fr_pixels / 2. <= f, f <= 2. * fr_pixels - 0.8 * fr_pixels / 2.), tf.complex64)
    fid = tf.reduce_sum(fid, [2, 3])
    fid = fid + tf.complex(tf.random.normal([2 * fr_pixels, series], 0., nd_value),
                           tf.random.normal([2 * fr_pixels, series], 0., nd_value))
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
        #Generated spectrum
        spectrum = get(ob_labels, sp_values, fr_values, dc_values, am_values, nd_value)
        yield spectrum, (ob_labels, sp_shifts, fr_shifts, dc_shifts, am_shifts)

#Normalizes spectra to zero mean and unit variance
@tf.function
def normalize(spectrum, target):
    spectrum = (spectrum - 179.24811) / 216.3278 #noise 0.0
#    spectrum = (spectrum - 180.86696) / 217.4606 #noise 0.1
#    spectrum = (spectrum - 179.58424) / 217.31755 #noise 0.2
#    spectrum = (spectrum - 179.61589) / 222.4815 #noise 0.5
#    spectrum = (spectrum - 181.52248) / 239.33989 #noise 1.0
#    spectrum = (spectrum - 179.96593) / 296.3282 #noise 2.0
#    spectrum = (spectrum - 179.81058) / 550.0122 #noise 5.0
    return spectrum, target

#Batched dataset of normalized spectra from the above generator
dataset = tf.data.Dataset.from_generator(generate,
                                         output_signature = (tf.TensorSpec([sp_pixels, fr_pixels]),
                                                             (tf.TensorSpec([sp_cells, fr_cells]),
                                                              tf.TensorSpec([sp_cells, fr_cells]),
                                                              tf.TensorSpec([sp_cells, fr_cells]),
                                                              tf.TensorSpec([sp_cells, fr_cells]),
                                                              tf.TensorSpec([sp_cells, fr_cells]))))
dataset = dataset.map(normalize).batch(65536)

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
ob_labels0 = np.where(ob_labels0 < 0.5, False, True)

#Predicted objectness labels
ob_labels1 = np.where(ob_logits1 < 0., False, True)

#Confusion matrix
confusion = sklearn.metrics.confusion_matrix(ob_labels0.flatten(), ob_labels1.flatten())

#Dataframe with various accuracies
labels = pd.DataFrame(columns = ['ob'], index = ['acc', 'neg', 'pos'])

#Total accuracy, true negatives and true positives
labels['ob']['acc'] = confusion.diagonal().sum() / confusion.sum()
labels['ob']['neg'] = confusion[0, 0] / confusion[0].sum()
labels['ob']['pos'] = confusion[1, 1] / confusion[1].sum()

#Save the dataframe with various accuracies
labels.to_csv('labels00.csv', float_format = '%.6e')

#Objectness labels for correctly predicted peaks
ob_labels = np.logical_and(ob_labels1, ob_labels0)

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

#Dataframe with various units errors
units = pd.DataFrame(columns = ['sp', 'fr', 'dc', 'am'], index = ['rmse', 'me', 'estd'])

#Root-mean-squared errors of units for correctly predicted peaks
units['sp']['rmse'] = np.sqrt(np.square(sp_units1 - sp_units0).mean())
units['fr']['rmse'] = np.sqrt(np.square(fr_units1 - fr_units0).mean())
units['dc']['rmse'] = np.sqrt(np.square(dc_units1 - dc_units0).mean())
units['am']['rmse'] = np.sqrt(np.square(am_units1 - am_units0).mean())

#Mean errors of units for correctly predicted peaks
units['sp']['me'] = np.mean(sp_units1 - sp_units0)
units['fr']['me'] = np.mean(fr_units1 - fr_units0)
units['dc']['me'] = np.mean(dc_units1 - dc_units0)
units['am']['me'] = np.mean(am_units1 - am_units0)

#Standard deviations of errors in units for correctly predicted peaks
units['sp']['estd'] = np.sqrt(np.square(sp_units1 - units['sp']['me'] - sp_units0).mean())
units['fr']['estd'] = np.sqrt(np.square(fr_units1 - units['fr']['me'] - fr_units0).mean())
units['dc']['estd'] = np.sqrt(np.square(dc_units1 - units['dc']['me'] - dc_units0).mean())
units['am']['estd'] = np.sqrt(np.square(am_units1 - units['am']['me'] - am_units0).mean())

#Save dataframe with various units errors
units.to_csv('units00.csv', float_format = '%.6e')
