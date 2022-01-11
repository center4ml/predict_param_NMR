import math
import tensorflow as tf

#Trains the YOLO model via an auxiliary trainer model.

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
        sp_shifts = tf.random.uniform([sp_cells, fr_cells]) * ob_labels
        fr_shifts = tf.random.uniform([sp_cells, fr_cells]) * ob_labels
        dc_shifts = tf.random.uniform([sp_cells, fr_cells]) * ob_labels
        am_shifts = tf.random.uniform([sp_cells, fr_cells]) * ob_labels
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
        #Spectrum
        spectrum = get(ob_labels, sp_values, fr_values, dc_values, am_values)
        yield (spectrum, ob_labels), (ob_labels, sp_shifts, fr_shifts, dc_shifts, am_shifts)

#Normalizes spectra to zero mean and unit variance
@tf.function
def normalize(data, target):
    spectrum, ob_labels = data
    spectrum = (spectrum - 180.45589) / 217.40448
    return (spectrum, ob_labels), target

#Batched dataset of normalized spectra from the above generator
dataset = tf.data.Dataset.from_generator(generate,
                                         output_signature = ((tf.TensorSpec([sp_pixels, fr_pixels]),
                                                              tf.TensorSpec([sp_cells, fr_cells])),
                                                             (tf.TensorSpec([sp_cells, fr_cells]),
                                                              tf.TensorSpec([sp_cells, fr_cells]),
                                                              tf.TensorSpec([sp_cells, fr_cells]),
                                                              tf.TensorSpec([sp_cells, fr_cells]),
                                                              tf.TensorSpec([sp_cells, fr_cells]))))
dataset = dataset.map(normalize).batch(64).prefetch(tf.data.experimental.AUTOTUNE)

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

#Model summary
model.summary()

#Auxiliary trainer model that sets peak shifts to zero in cells without a peak
spectrum = tf.keras.Input([256, 256])
ob_labels = tf.keras.Input([16, 16])
ob_logits, sp_shifts, fr_shifts, dc_shifts, am_shifts = model(spectrum)
ob_logits = tf.keras.layers.Lambda(lambda logits: logits, name = 'ob')(ob_logits)
sp_shifts = sp_shifts * ob_labels
sp_shifts = tf.keras.layers.Lambda(lambda shifts: shifts, name = 'sp')(sp_shifts)
fr_shifts = fr_shifts * ob_labels
fr_shifts = tf.keras.layers.Lambda(lambda shifts: shifts, name = 'fr')(fr_shifts)
dc_shifts = dc_shifts * ob_labels
dc_shifts = tf.keras.layers.Lambda(lambda shifts: shifts, name = 'dc')(dc_shifts)
am_shifts = am_shifts * ob_labels
am_shifts = tf.keras.layers.Lambda(lambda shifts: shifts, name = 'am')(am_shifts)
trainer = tf.keras.models.Model((spectrum, ob_labels), (ob_logits, sp_shifts, fr_shifts, dc_shifts, am_shifts))

#Optimizer, loss functions and metrics
trainer.compile(tf.keras.optimizers.Adam(),
                {'ob': tf.keras.losses.BinaryCrossentropy(from_logits = True),
                 'sp': tf.keras.losses.MeanSquaredError(),
                 'fr': tf.keras.losses.MeanSquaredError(),
                 'dc': tf.keras.losses.MeanSquaredError(),
                 'am': tf.keras.losses.MeanSquaredError()},
                {'ob': tf.keras.metrics.BinaryAccuracy('acc', threshold = 0.),
                 'sp': tf.keras.metrics.RootMeanSquaredError('rms'),
                 'fr': tf.keras.metrics.RootMeanSquaredError('rms'),
                 'dc': tf.keras.metrics.RootMeanSquaredError('rms'),
                 'am': tf.keras.metrics.RootMeanSquaredError('rms')})

#Train the actual model via the trainer model and saving weights of the trainer model
trainer.fit(dataset, steps_per_epoch = 1, epochs = 262144,
            callbacks = [tf.keras.callbacks.CSVLogger('04train.csv', ' '),
                         tf.keras.callbacks.ModelCheckpoint('04train.h5',
                                                          monitor = 'loss',
                                                          verbose = 1,
                                                          save_best_only = True,
                                                          save_weights_only = True)])
