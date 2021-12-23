import math
import tensorflow as tf

#Calculates the mean and standard deviation of the generated spectra. They will be used for normalizing the spectra passed to the model.

#Slightly modified constants from Daniel
series = 20
dwmin, dwmax = -10., 10.

#Ranges of speed, frequency, damping coefficient and amplitude in units used in experiment
sp_min, sp_max = -10., 10.
fr_min, fr_max = 0., 256.
dc_min, dc_max = 0.1, 10.
am_min, am_max = 0.1, 5.

#Number of pixels along the speed and frequency dimension
sp_pixels, fr_pixels = 256, 256

#Auxiliary arrays for generating spectrum in tensorflow
t = tf.complex(tf.range(2 * fr_pixels, dtype = tf.float32) / fr_pixels / 2., 0.)
a = tf.range(series, dtype = tf.float32)
b = 2. * math.pi * \
    tf.complex(0., tf.range(dwmin, dwmax, (dwmax - dwmin) / sp_pixels, dtype = tf.float32))[:, None] * t

#Takes peak parameters and returns the resulting spectrum
def get(sp_values, fr_values, dc_values, am_values):
    f = a[:, None] * sp_values + fr_values + fr_pixels / 2.
    fid = tf.complex(am_values, 0.) * tf.exp(2. * math.pi * t[:, None, None] * (tf.complex(-dc_values, f))) * \
          tf.cast(tf.logical_and(0.8 * fr_pixels / 2. <= f, f <= 2. * fr_pixels - 0.8 * fr_pixels / 2.), tf.complex64)
    fid = tf.reduce_sum(fid, 2)
    fid = tf.concat([fid[: 1] / 2., fid[1:]], 0)
    p = tf.exp(-b[:, :, None] * tf.complex(a, 0.)) * fid
    return tf.math.real(tf.signal.fft(tf.reduce_sum(p, 2)))[:, fr_pixels // 2 : 2 * fr_pixels - fr_pixels // 2]

#Draws random peak parameters and yields the resulting spectrum
def generate():
    while True:
        #Random number of peaks from 1 to 9
        count = tf.random.uniform([], 1, 10, dtype = tf.int32)
        #Random peak parameters in units used in experiment
        sp_values = tf.random.uniform([count], sp_min, sp_max)
        fr_values = tf.random.uniform([count], fr_min, fr_max)
        dc_values = tf.random.uniform([count], dc_min, dc_max)
        am_values = tf.random.uniform([count], am_min, am_max)
        #Spectrum resulting from these parameters
        spectrum = get(sp_values, fr_values, dc_values, am_values)
        yield spectrum

#Batched dataset of spectra from the above generator
dataset = tf.data.Dataset.from_generator(generate,
                                         output_signature = tf.TensorSpec([sp_pixels, fr_pixels]))
dataset = dataset.batch(4096)

#Mean and standard deviation of spectra
spectrum = next(iter(dataset))
print('spectrum:', tf.math.reduce_mean(spectrum).numpy(), tf.math.reduce_std(spectrum).numpy())

#Resulting mean and standard deviation used later for normalization
#spectrum: 180.45589 217.40448
