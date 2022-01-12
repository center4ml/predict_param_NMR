A fully convolutional YOLO model is developed that takes a spectrum containing 1 to 9 peaks and returns the speed, frequency, damping coefficient and amplitude of each peak. 256 x 256 spectra with 16 x 16 YOLO grid are assumed. The training spectra are generated without aliasing, without multiplets, but with noise. Complex gaussian noise is added to the FID and various noise amplitudes are considered.

01test.py: Demonstrates that spectra generated in tensorflow are identical to those generated in numpy. Randomly takes 1 to 9 peaks and distributes them at random regardless of the YOLO grid. Displays spectra generated in tensorflow and numpy. Calculates mean squared differences between them.

02normalize.py: Calculates the mean and standard deviation of the generated spectra. They will be used for normalizing the spectra passed to the model. The normalization is calculated separately for each noise amplitude.

03generate.py: Generates spectra in tensorflow for use with the YOLO model. Randomly takes 1 to 9 peaks and distributes them at random so that each cell contains at most one peak. Displays the spectra.

04train.py: Trains the YOLO model via an auxiliary trainer model.

05extract.py: Extracts the parameters of the YOLO model from the auxiliary trainer model.

06demo.py: Plots a random spectrum with generated and predicted peaks.

07evaluate.py: Evaluates the model performance on a large test batch. Calculates objectness accuracy. Calculates root-mean-square errors, mean errors and error standard deviations of the predicted peak parameters.

08plot.py: Plots various accuracies and units errors versus noise amplitude

09histogram.py: Makes histograms of errors in the predicted peak parameters for a given noise amplitude. Displays the resulting plots.
