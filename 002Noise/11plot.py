from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

acc_am00 = pd.read_csv('acc_am00.csv')
acc_dc00 = pd.read_csv('acc_dc00.csv')
acc_am01 = pd.read_csv('acc_am01.csv')
acc_dc01 = pd.read_csv('acc_dc01.csv')
acc_am02 = pd.read_csv('acc_am02.csv')
acc_dc02 = pd.read_csv('acc_dc02.csv')
acc_am05 = pd.read_csv('acc_am05.csv')
acc_dc05 = pd.read_csv('acc_dc05.csv')
acc_am10 = pd.read_csv('acc_am10.csv')
acc_dc10 = pd.read_csv('acc_dc10.csv')
acc_am20 = pd.read_csv('acc_am20.csv')
acc_dc20 = pd.read_csv('acc_dc20.csv')
acc_am50 = pd.read_csv('acc_am50.csv')
acc_dc50 = pd.read_csv('acc_dc50.csv')

plt.figure(figsize = (6, 3), tight_layout = True)
plt.suptitle('Accuracy versus units')

plt.subplot(1, 2, 1)
plt.plot(acc_am00['units'], acc_am00['ob'])
plt.plot(acc_am01['units'], acc_am01['ob'])
plt.plot(acc_am02['units'], acc_am02['ob'])
plt.plot(acc_am05['units'], acc_am05['ob'])
plt.plot(acc_am10['units'], acc_am10['ob'])
plt.plot(acc_am20['units'], acc_am20['ob'])
plt.plot(acc_am50['units'], acc_am50['ob'])
plt.grid()
plt.ylim(0.0, 1.0)
plt.xlim(0., 1.)
plt.ylabel('Objectness')
plt.xlabel('Amplitude')

plt.subplot(1, 2, 2)
plt.plot(acc_dc00['units'], acc_dc00['ob'])
plt.plot(acc_dc01['units'], acc_dc01['ob'])
plt.plot(acc_dc02['units'], acc_dc02['ob'])
plt.plot(acc_dc05['units'], acc_dc05['ob'])
plt.plot(acc_dc10['units'], acc_dc10['ob'])
plt.plot(acc_dc20['units'], acc_dc20['ob'])
plt.plot(acc_dc50['units'], acc_dc50['ob'])
plt.grid()
plt.ylim(0.0, 1.0)
plt.xlim(0., 1.)
plt.xlabel('Damping coefficient')


rmse_am00 = pd.read_csv('rmse_am00.csv')
rmse_dc00 = pd.read_csv('rmse_dc00.csv')
rmse_am01 = pd.read_csv('rmse_am01.csv')
rmse_dc01 = pd.read_csv('rmse_dc01.csv')
rmse_am02 = pd.read_csv('rmse_am02.csv')
rmse_dc02 = pd.read_csv('rmse_dc02.csv')
rmse_am05 = pd.read_csv('rmse_am05.csv')
rmse_dc05 = pd.read_csv('rmse_dc05.csv')
rmse_am10 = pd.read_csv('rmse_am10.csv')
rmse_dc10 = pd.read_csv('rmse_dc10.csv')
rmse_am20 = pd.read_csv('rmse_am20.csv')
rmse_dc20 = pd.read_csv('rmse_dc20.csv')
rmse_am50 = pd.read_csv('rmse_am50.csv')
rmse_dc50 = pd.read_csv('rmse_dc50.csv')

plt.figure(figsize = (6, 8), tight_layout = True)
plt.suptitle('Units RMSE versus units')

#Speed units root-mean-square error in amplitude strata
plt.subplot(4, 2, 1)
plt.plot(rmse_am00['units'], rmse_am00['sp'])
plt.plot(rmse_am01['units'], rmse_am01['sp'])
plt.plot(rmse_am02['units'], rmse_am02['sp'])
plt.plot(rmse_am05['units'], rmse_am05['sp'])
plt.plot(rmse_am10['units'], rmse_am10['sp'])
plt.plot(rmse_am20['units'], rmse_am20['sp'])
plt.plot(rmse_am50['units'], rmse_am50['sp'])
plt.grid()
plt.ylim(0.003, 0.013)
plt.xlim(0., 1.)
plt.ylabel('Speed')

#Speed units root-mean-square error in damping strata
plt.subplot(4, 2, 2)
plt.plot(rmse_dc00['units'], rmse_dc00['sp'])
plt.plot(rmse_dc01['units'], rmse_dc01['sp'])
plt.plot(rmse_dc02['units'], rmse_dc02['sp'])
plt.plot(rmse_dc05['units'], rmse_dc05['sp'])
plt.plot(rmse_dc10['units'], rmse_dc10['sp'])
plt.plot(rmse_dc20['units'], rmse_dc20['sp'])
plt.plot(rmse_dc50['units'], rmse_dc50['sp'])
plt.grid()
plt.ylim(0.003, 0.013)
plt.xlim(0., 1.)

#Frequency units root-mean-square error in amplitude strata
plt.subplot(4, 2, 3)
plt.plot(rmse_am00['units'], rmse_am00['fr'])
plt.plot(rmse_am01['units'], rmse_am01['fr'])
plt.plot(rmse_am02['units'], rmse_am02['fr'])
plt.plot(rmse_am05['units'], rmse_am05['fr'])
plt.plot(rmse_am10['units'], rmse_am10['fr'])
plt.plot(rmse_am20['units'], rmse_am20['fr'])
plt.plot(rmse_am50['units'], rmse_am50['fr'])
plt.grid()
plt.ylim(0.002, 0.012)
plt.xlim(0., 1.)
plt.ylabel('Frequency')

#Frequency units root-mean-square error in damping strata
plt.subplot(4, 2, 4)
plt.plot(rmse_dc00['units'], rmse_dc00['fr'])
plt.plot(rmse_dc01['units'], rmse_dc01['fr'])
plt.plot(rmse_dc02['units'], rmse_dc02['fr'])
plt.plot(rmse_dc05['units'], rmse_dc05['fr'])
plt.plot(rmse_dc10['units'], rmse_dc10['fr'])
plt.plot(rmse_dc20['units'], rmse_dc20['fr'])
plt.plot(rmse_dc50['units'], rmse_dc50['fr'])
plt.grid()
plt.ylim(0.002, 0.012)
plt.xlim(0., 1.)

#Damping units root-mean-square error in amplitude strata
plt.subplot(4, 2, 5)
plt.plot(rmse_am00['units'], rmse_am00['dc'])
plt.plot(rmse_am01['units'], rmse_am01['dc'])
plt.plot(rmse_am02['units'], rmse_am02['dc'])
plt.plot(rmse_am05['units'], rmse_am05['dc'])
plt.plot(rmse_am10['units'], rmse_am10['dc'])
plt.plot(rmse_am20['units'], rmse_am20['dc'])
plt.plot(rmse_am50['units'], rmse_am50['dc'])
plt.grid()
plt.ylim(0.00, 0.25)
plt.xlim(0., 1.)
plt.ylabel('Damping coefficient')

#Damping units root-mean-square error in damping strata
plt.subplot(4, 2, 6)
plt.plot(rmse_dc00['units'], rmse_dc00['dc'])
plt.plot(rmse_dc01['units'], rmse_dc01['dc'])
plt.plot(rmse_dc02['units'], rmse_dc02['dc'])
plt.plot(rmse_dc05['units'], rmse_dc05['dc'])
plt.plot(rmse_dc10['units'], rmse_dc10['dc'])
plt.plot(rmse_dc20['units'], rmse_dc20['dc'])
plt.plot(rmse_dc50['units'], rmse_dc50['dc'])
plt.grid()
plt.ylim(0.00, 0.25)
plt.xlim(0., 1.)

#Amplitude units root-mean-square error in amplitude strata
plt.subplot(4, 2, 7)
plt.plot(rmse_am00['units'], rmse_am00['am'])
plt.plot(rmse_am01['units'], rmse_am01['am'])
plt.plot(rmse_am02['units'], rmse_am02['am'])
plt.plot(rmse_am05['units'], rmse_am05['am'])
plt.plot(rmse_am10['units'], rmse_am10['am'])
plt.plot(rmse_am20['units'], rmse_am20['am'])
plt.plot(rmse_am50['units'], rmse_am50['am'])
plt.grid()
plt.ylim(0.00, 0.25)
plt.xlim(0., 1.)
plt.ylabel('Amplitude')
plt.xlabel('Amplitude')

#Amplitude units root-mean-square error in damping strata
plt.subplot(4, 2, 8)
plt.plot(rmse_dc00['units'], rmse_dc00['am'])
plt.plot(rmse_dc01['units'], rmse_dc01['am'])
plt.plot(rmse_dc02['units'], rmse_dc02['am'])
plt.plot(rmse_dc05['units'], rmse_dc05['am'])
plt.plot(rmse_dc10['units'], rmse_dc10['am'])
plt.plot(rmse_dc20['units'], rmse_dc20['am'])
plt.plot(rmse_dc50['units'], rmse_dc50['am'])
plt.grid()
plt.ylim(0.00, 0.25)
plt.xlim(0., 1.)
plt.xlabel('Damping coefficient')



me_am00 = pd.read_csv('me_am00.csv')
me_dc00 = pd.read_csv('me_dc00.csv')
me_am01 = pd.read_csv('me_am01.csv')
me_dc01 = pd.read_csv('me_dc01.csv')
me_am02 = pd.read_csv('me_am02.csv')
me_dc02 = pd.read_csv('me_dc02.csv')
me_am05 = pd.read_csv('me_am05.csv')
me_dc05 = pd.read_csv('me_dc05.csv')
me_am10 = pd.read_csv('me_am10.csv')
me_dc10 = pd.read_csv('me_dc10.csv')
me_am20 = pd.read_csv('me_am20.csv')
me_dc20 = pd.read_csv('me_dc20.csv')
me_am50 = pd.read_csv('me_am50.csv')
me_dc50 = pd.read_csv('me_dc50.csv')

plt.figure(figsize = (6, 8), tight_layout = True)
plt.suptitle('Units ME versus units')

#Speed units mean error in amplitude strata
plt.subplot(4, 2, 1)
plt.plot(me_am00['units'], me_am00['sp'])
plt.plot(me_am01['units'], me_am01['sp'])
plt.plot(me_am02['units'], me_am02['sp'])
plt.plot(me_am05['units'], me_am05['sp'])
plt.plot(me_am10['units'], me_am10['sp'])
plt.plot(me_am20['units'], me_am20['sp'])
plt.plot(me_am50['units'], me_am50['sp'])
plt.grid()
plt.xlim(0., 1.)
plt.ylabel('Speed')

#Speed units mean error in damping strata
plt.subplot(4, 2, 2)
plt.plot(me_dc00['units'], me_dc00['sp'])
plt.plot(me_dc01['units'], me_dc01['sp'])
plt.plot(me_dc02['units'], me_dc02['sp'])
plt.plot(me_dc05['units'], me_dc05['sp'])
plt.plot(me_dc10['units'], me_dc10['sp'])
plt.plot(me_dc20['units'], me_dc20['sp'])
plt.plot(me_dc50['units'], me_dc50['sp'])
plt.grid()
plt.xlim(0., 1.)

#Frequency units mean error in amplitude strata
plt.subplot(4, 2, 3)
plt.plot(me_am00['units'], me_am00['fr'])
plt.plot(me_am01['units'], me_am01['fr'])
plt.plot(me_am02['units'], me_am02['fr'])
plt.plot(me_am05['units'], me_am05['fr'])
plt.plot(me_am10['units'], me_am10['fr'])
plt.plot(me_am20['units'], me_am20['fr'])
plt.plot(me_am50['units'], me_am50['fr'])
plt.grid()
plt.xlim(0., 1.)
plt.ylabel('Frequency')

#Frequency units mean error in damping strata
plt.subplot(4, 2, 4)
plt.plot(me_dc00['units'], me_dc00['fr'])
plt.plot(me_dc01['units'], me_dc01['fr'])
plt.plot(me_dc02['units'], me_dc02['fr'])
plt.plot(me_dc05['units'], me_dc05['fr'])
plt.plot(me_dc10['units'], me_dc10['fr'])
plt.plot(me_dc20['units'], me_dc20['fr'])
plt.plot(me_dc50['units'], me_dc50['fr'])
plt.grid()
plt.xlim(0., 1.)

#Damping units mean error in amplitude strata
plt.subplot(4, 2, 5)
plt.plot(me_am00['units'], me_am00['dc'])
plt.plot(me_am01['units'], me_am01['dc'])
plt.plot(me_am02['units'], me_am02['dc'])
plt.plot(me_am05['units'], me_am05['dc'])
plt.plot(me_am10['units'], me_am10['dc'])
plt.plot(me_am20['units'], me_am20['dc'])
plt.plot(me_am50['units'], me_am50['dc'])
plt.grid()
plt.xlim(0., 1.)
plt.ylabel('Damping coefficient')

#Damping units mean error in damping strata
plt.subplot(4, 2, 6)
plt.plot(me_dc00['units'], me_dc00['dc'])
plt.plot(me_dc01['units'], me_dc01['dc'])
plt.plot(me_dc02['units'], me_dc02['dc'])
plt.plot(me_dc05['units'], me_dc05['dc'])
plt.plot(me_dc10['units'], me_dc10['dc'])
plt.plot(me_dc20['units'], me_dc20['dc'])
plt.plot(me_dc50['units'], me_dc50['dc'])
plt.grid()
plt.xlim(0., 1.)

#Amplitude units mean error in amplitude strata
plt.subplot(4, 2, 7)
plt.plot(me_am00['units'], me_am00['am'])
plt.plot(me_am01['units'], me_am01['am'])
plt.plot(me_am02['units'], me_am02['am'])
plt.plot(me_am05['units'], me_am05['am'])
plt.plot(me_am10['units'], me_am10['am'])
plt.plot(me_am20['units'], me_am20['am'])
plt.plot(me_am50['units'], me_am50['am'])
plt.grid()
plt.xlim(0., 1.)
plt.ylabel('Amplitude')
plt.xlabel('Amplitude')

#Amplitude units mean error in damping strata
plt.subplot(4, 2, 8)
plt.plot(me_dc00['units'], me_dc00['am'])
plt.plot(me_dc01['units'], me_dc01['am'])
plt.plot(me_dc02['units'], me_dc02['am'])
plt.plot(me_dc05['units'], me_dc05['am'])
plt.plot(me_dc10['units'], me_dc10['am'])
plt.plot(me_dc20['units'], me_dc20['am'])
plt.plot(me_dc50['units'], me_dc50['am'])
plt.grid()
plt.xlim(0., 1.)
plt.xlabel('Damping coefficient')





estd_am00 = pd.read_csv('estd_am00.csv')
estd_dc00 = pd.read_csv('estd_dc00.csv')
estd_am01 = pd.read_csv('estd_am01.csv')
estd_dc01 = pd.read_csv('estd_dc01.csv')
estd_am02 = pd.read_csv('estd_am02.csv')
estd_dc02 = pd.read_csv('estd_dc02.csv')
estd_am05 = pd.read_csv('estd_am05.csv')
estd_dc05 = pd.read_csv('estd_dc05.csv')
estd_am10 = pd.read_csv('estd_am10.csv')
estd_dc10 = pd.read_csv('estd_dc10.csv')
estd_am20 = pd.read_csv('estd_am20.csv')
estd_dc20 = pd.read_csv('estd_dc20.csv')
estd_am50 = pd.read_csv('estd_am50.csv')
estd_dc50 = pd.read_csv('estd_dc50.csv')

plt.figure(figsize = (6, 8), tight_layout = True)
plt.suptitle('Units ESTD versus units')

#Speed units error standard deviation in amplitude strata
plt.subplot(4, 2, 1)
plt.plot(estd_am00['units'], estd_am00['sp'])
plt.plot(estd_am01['units'], estd_am01['sp'])
plt.plot(estd_am02['units'], estd_am02['sp'])
plt.plot(estd_am05['units'], estd_am05['sp'])
plt.plot(estd_am10['units'], estd_am10['sp'])
plt.plot(estd_am20['units'], estd_am20['sp'])
plt.plot(estd_am50['units'], estd_am50['sp'])
plt.grid()
plt.ylim(0.003, 0.013)
plt.xlim(0., 1.)
plt.ylabel('Speed')

#Speed units error standard deviation in damping strata
plt.subplot(4, 2, 2)
plt.plot(estd_dc00['units'], estd_dc00['sp'])
plt.plot(estd_dc01['units'], estd_dc01['sp'])
plt.plot(estd_dc02['units'], estd_dc02['sp'])
plt.plot(estd_dc05['units'], estd_dc05['sp'])
plt.plot(estd_dc10['units'], estd_dc10['sp'])
plt.plot(estd_dc20['units'], estd_dc20['sp'])
plt.plot(estd_dc50['units'], estd_dc50['sp'])
plt.grid()
plt.ylim(0.003, 0.013)
plt.xlim(0., 1.)

#Frequency units error standard deviation in amplitude strata
plt.subplot(4, 2, 3)
plt.plot(estd_am00['units'], estd_am00['fr'])
plt.plot(estd_am01['units'], estd_am01['fr'])
plt.plot(estd_am02['units'], estd_am02['fr'])
plt.plot(estd_am05['units'], estd_am05['fr'])
plt.plot(estd_am10['units'], estd_am10['fr'])
plt.plot(estd_am20['units'], estd_am20['fr'])
plt.plot(estd_am50['units'], estd_am50['fr'])
plt.grid()
plt.ylim(0.002, 0.012)
plt.xlim(0., 1.)
plt.ylabel('Frequency')

#Frequency units error standard deviation in damping strata
plt.subplot(4, 2, 4)
plt.plot(estd_dc00['units'], estd_dc00['fr'])
plt.plot(estd_dc01['units'], estd_dc01['fr'])
plt.plot(estd_dc02['units'], estd_dc02['fr'])
plt.plot(estd_dc05['units'], estd_dc05['fr'])
plt.plot(estd_dc10['units'], estd_dc10['fr'])
plt.plot(estd_dc20['units'], estd_dc20['fr'])
plt.plot(estd_dc50['units'], estd_dc50['fr'])
plt.grid()
plt.ylim(0.002, 0.012)
plt.xlim(0., 1.)

#Damping units error standard deviation in amplitude strata
plt.subplot(4, 2, 5)
plt.plot(estd_am00['units'], estd_am00['dc'])
plt.plot(estd_am01['units'], estd_am01['dc'])
plt.plot(estd_am02['units'], estd_am02['dc'])
plt.plot(estd_am05['units'], estd_am05['dc'])
plt.plot(estd_am10['units'], estd_am10['dc'])
plt.plot(estd_am20['units'], estd_am20['dc'])
plt.plot(estd_am50['units'], estd_am50['dc'])
plt.grid()
plt.ylim(0.00, 0.25)
plt.xlim(0., 1.)
plt.ylabel('Damping coefficient')

#Damping units error standard deviation in damping strata
plt.subplot(4, 2, 6)
plt.plot(estd_dc00['units'], estd_dc00['dc'])
plt.plot(estd_dc01['units'], estd_dc01['dc'])
plt.plot(estd_dc02['units'], estd_dc02['dc'])
plt.plot(estd_dc05['units'], estd_dc05['dc'])
plt.plot(estd_dc10['units'], estd_dc10['dc'])
plt.plot(estd_dc20['units'], estd_dc20['dc'])
plt.plot(estd_dc50['units'], estd_dc50['dc'])
plt.grid()
plt.ylim(0.00, 0.25)
plt.xlim(0., 1.)

#Amplitude units error standard deviation in amplitude strata
plt.subplot(4, 2, 7)
plt.plot(estd_am00['units'], estd_am00['am'])
plt.plot(estd_am01['units'], estd_am01['am'])
plt.plot(estd_am02['units'], estd_am02['am'])
plt.plot(estd_am05['units'], estd_am05['am'])
plt.plot(estd_am10['units'], estd_am10['am'])
plt.plot(estd_am20['units'], estd_am20['am'])
plt.plot(estd_am50['units'], estd_am50['am'])
plt.grid()
plt.ylim(0.00, 0.25)
plt.xlim(0., 1.)
plt.ylabel('Amplitude')
plt.xlabel('Amplitude')

#Amplitude units error standard deviation in damping strata
plt.subplot(4, 2, 8)
plt.plot(estd_dc00['units'], estd_dc00['am'])
plt.plot(estd_dc01['units'], estd_dc01['am'])
plt.plot(estd_dc02['units'], estd_dc02['am'])
plt.plot(estd_dc05['units'], estd_dc05['am'])
plt.plot(estd_dc10['units'], estd_dc10['am'])
plt.plot(estd_dc20['units'], estd_dc20['am'])
plt.plot(estd_dc50['units'], estd_dc50['am'])
plt.grid()
plt.ylim(0.00, 0.25)
plt.xlim(0., 1.)
plt.xlabel('Damping coefficient')



plt.show()