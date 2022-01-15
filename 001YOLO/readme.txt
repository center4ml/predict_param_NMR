A fully convolutional YOLO model is developed that takes a spectrum containing 1 to 9 peaks and returns the speed, frequency, damping coefficient and amplitude of each peak. 256 x 256 spectra with 16 x 16 YOLO grid are assumed. The training spectra are generated without aliasing, without multiplets, and without noise.

Subsequent steps:

01test.py: Demonstrates that spectra generated in tensorflow and numpy are identical to those generated in the original python code from Daniel. Randomly takes 1 to 9 peaks and distributes them at random regardless of the YOLO grid. Displays spectra generated in tensorflow, numpy and in the original code. Calculates mean squared differences between these spectra.

02normalize.py: Calculates the mean and standard deviation of the generated spectra. They will be used for normalizing the spectra passed to the model.

03generate.py: Generates spectra in tensorflow for use with the YOLO model. Randomly takes 1 to 9 peaks and distributes them at random so that each cell contains at most one peak. Displays the spectra.

04train.py: Trains the YOLO model via an auxiliary trainer model.

05extract.py: Extracts the parameters of the YOLO model from the auxiliary trainer model.

06demo.py: Plots a random spectrum with generated and predicted peaks.

07evaluate.py: Evaluates the model performance on a large test batch. Calculates objectness accuracy. Calculates root-mean-square errors, mean errors and error standard deviations of the predicted peak parameters.

08histogram.py: Makes histograms of errors in the predicted peak parameters and compares them against gaussian distributions. Displays the resulting plots.
