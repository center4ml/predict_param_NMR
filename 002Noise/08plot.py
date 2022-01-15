from matplotlib import pyplot as plt
import pandas as pd

#Plots various accuracies and units errors versus noise amplitude

#Noise amplitudes
noise = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

#Names of dataframes containing various accuracies for subsequent noise amplitudes
names = ['labels00', 'labels01', 'labels02', 'labels05', 'labels10', 'labels20', 'labels50']

#Various accuracies for subsequent noise amplitudes
ob_acc = list()
ob_neg = list()
ob_pos = list()
for name in names:
    labels = pd.read_csv(name + '.csv', index_col = 0)
    ob_acc.append(labels['ob']['acc'])
    ob_neg.append(labels['ob']['neg'])
    ob_pos.append(labels['ob']['pos'])

#Plot various accuracies versus noise amplitude
plt.figure(figsize = (4, 6), tight_layout = True)
plt.suptitle('Accuracy versus values')
plt.subplot(3, 1, 1)
plt.plot(noise, ob_acc)
plt.grid()
plt.ylim(0.985, 1.000)
plt.xlim(0., 5.)
plt.ylabel('Total')
plt.subplot(3, 1, 2)
plt.plot(noise, ob_neg)
plt.grid()
plt.ylim(0.985, 1.000)
plt.xlim(0., 5.)
plt.ylabel('Negative')
plt.subplot(3, 1, 3)
plt.plot(noise, ob_pos)
plt.grid()
plt.ylim(0.4, 1.000)
plt.xlim(0., 5.)
plt.ylabel('Positive')
plt.xlabel('Noise amplitude')

#Names of dataframes containing various units errors for subsequent noise amplitudes
names = ['units00', 'units01', 'units02', 'units05', 'units10', 'units20', 'units50']

#Root-mean-square errors for subsequent noise amplitudes
sp_rmse, fr_rmse, dc_rmse, am_rmse = list(), list(), list(), list()
for name in names:
    units = pd.read_csv(name + '.csv', index_col = 0)
    sp_rmse.append(units['sp']['rmse'])
    fr_rmse.append(units['fr']['rmse'])
    dc_rmse.append(units['dc']['rmse'])
    am_rmse.append(units['am']['rmse'])

#Plot root-mean-square errors versus noise amplitude
plt.figure(figsize = (4, 8), tight_layout = True)
plt.suptitle('Units RMSE versus values')
plt.subplot(4, 1, 1)
plt.plot(noise, sp_rmse)
plt.grid()
plt.xlim(0., 5.)
plt.ylabel('Speed')
plt.subplot(4, 1, 2)
plt.plot(noise, fr_rmse)
plt.grid()
plt.xlim(0., 5.)
plt.ylabel('Frequency')
plt.subplot(4, 1, 3)
plt.plot(noise, dc_rmse)
plt.grid()
plt.xlim(0., 5.)
plt.ylabel('Damping')
plt.subplot(4, 1, 4)
plt.plot(noise, am_rmse)
plt.grid()
plt.xlim(0., 5.)
plt.ylabel('Amplitude')
plt.xlabel('Noise amplitude')

#Mean errors for subsequent noise amplitudes
sp_me, fr_me, dc_me, am_me = list(), list(), list(), list()
for name in names:
    units = pd.read_csv(name + '.csv', index_col = 0)
    sp_me.append(units['sp']['me'])
    fr_me.append(units['fr']['me'])
    dc_me.append(units['dc']['me'])
    am_me.append(units['am']['me'])

#Plot mean errors versus noise amplitude
plt.figure(figsize = (4, 8), tight_layout = True)
plt.suptitle('Units ME versus values')
plt.subplot(4, 1, 1)
plt.plot(noise, sp_me)
plt.grid()
plt.xlim(0., 5.)
plt.ylabel('Speed')
plt.subplot(4, 1, 2)
plt.plot(noise, fr_me)
plt.grid()
plt.xlim(0., 5.)
plt.ylabel('Frequency')
plt.subplot(4, 1, 3)
plt.plot(noise, dc_me)
plt.grid()
plt.xlim(0., 5.)
plt.ylabel('Damping')
plt.subplot(4, 1, 4)
plt.plot(noise, am_me)
plt.grid()
plt.xlim(0., 5.)
plt.ylabel('Amplitude')
plt.xlabel('Noise amplirude')

#Error standard deviations for subsequent noise amplitudes
sp_estd, fr_estd, dc_estd, am_estd = list(), list(), list(), list()
for name in names:
    units = pd.read_csv(name + '.csv', index_col = 0)
    sp_estd.append(units['sp']['estd'])
    fr_estd.append(units['fr']['estd'])
    dc_estd.append(units['dc']['estd'])
    am_estd.append(units['am']['estd'])

#Plot error standard deviations versus noise amplitude
plt.figure(figsize = (4, 8), tight_layout = True)
plt.suptitle('Units ESTD versus values')
plt.subplot(4, 1, 1)
plt.plot(noise, sp_estd)
plt.grid()
plt.xlim(0., 5.)
plt.ylabel('Speed')
plt.subplot(4, 1, 2)
plt.plot(noise, fr_estd)
plt.grid()
plt.xlim(0., 5.)
plt.ylabel('Frequency')
plt.subplot(4, 1, 3)
plt.plot(noise, dc_estd)
plt.grid()
plt.xlim(0., 5.)
plt.ylabel('Damping')
plt.subplot(4, 1, 4)
plt.plot(noise, am_estd)
plt.grid()
plt.xlim(0., 5.)
plt.ylabel('Amplitude')
plt.xlabel('Noise amplitude')

plt.show()
