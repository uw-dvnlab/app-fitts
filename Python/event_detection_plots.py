# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 15:06:08 2023

@author: zafar
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt
from scipy.ndimage import label

# Read
trial_no = 1
df_ah = pd.read_csv(f'./fitts_ah_{trial_no}.csv')
df_ot = pd.read_csv(f'./fitts_ot_{trial_no}.csv')
df_ot = df_ot.interpolate(method='cubic')
lbl_sys = ['ah', 'ot']

#%
FS = [30, 250]
X = [df_ah.x, df_ot.x]
T = [df_ah.time, df_ot.time]

f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)

plots = [ax1, ax2, ax3, ax4]

for p in range(len(plots)):
    if p<len(X):
        x = X[p]
        t = T[p]
        #% Interpolate data
        FSi = 250
        tI = np.arange(1/FSi, max(t), 1/FSi)
        cs = CubicSpline(t, x)
        xI = cs(tI)
        vI = np.gradient(xI) * FSi
        
        # plt.plot(tI, vI)
        # plt.plot(t, v)
        
        #% Filter
        FC = 10
        WN = FC / (FSi/2)
        b, a = butter(2, WN)
        vI = filtfilt(b, a, vI)
        
        # plt.plot(t, v)
        # plt.plot(tI, vI)
        
        #% Find thresholds
        # Movement onset: >20mm/s
        v_on_20 = np.abs(vI.copy())
        v_on_20[v_on_20<20] = 0
        
        lbl, num_lbl = label(v_on_20)
        move_on_20 = []
        move_off_20 = []
        for i in range(num_lbl):
            idx = np.where(lbl==(i+1))[0]
            move_time = idx[-1] - idx[0]
            if move_time > 50:
                move_on_20.append(idx[0])
                move_off_20.append(idx[-1])
        
        # Movement onset: >100mm/s
        v_on_100 = np.abs(vI.copy())
        v_on_100[v_on_100<100] = 0
        
        lbl, num_lbl = label(v_on_100)
        move_off_100 = []
        for i in range(num_lbl):
            idx = np.where(lbl==(i+1))[0]
            move_time = idx[-1] - idx[0]
            if move_time > 50:
                move_off_100.append(idx[-1])
        
        # Peak velocities
        pk_vel, _ = find_peaks(np.abs(vI), height=500, distance=20)
        
        plots[p].plot(tI, xI)
        plots[p].vlines(tI[move_on_20], -125, 125, color='g')
        plots[p].plot(tI[move_on_20], xI[move_on_20], 'go')
        plots[p].plot(tI[move_off_20], xI[move_off_20], 'ro')
        plots[p].plot(tI[move_off_100], xI[move_off_100], 'mo')
        plots[p].plot(tI[pk_vel], xI[pk_vel], 'ko')
        plots[p].set_xlim(0, 4)
        plots[p].set_title(f'{lbl_sys[p]}')
    else:
        x = X[p-2]
        t = T[p-2]
        #% Interpolate data
        FSi = 250
        tI = np.arange(1/FSi, max(t), 1/FSi)
        cs = CubicSpline(t, x)
        xI = cs(tI)
        vI = np.gradient(xI) * FSi
        
        #% Filter
        FC = 10
        WN = FC / (FSi/2)
        b, a = butter(2, WN)
        vI = filtfilt(b, a, vI)
        
        plots[2].plot(tI, xI, label=lbl_sys[p-2])
        plots[3].plot(tI, vI, label=lbl_sys[p-2])
        plots[2].set_xlim(0, 4)
        plots[3].set_xlim(0, 4)
        plots[2].legend()
        plots[3].legend()
        plots[2].set_title('position')
        plots[3].set_title('velocity')
    
#%%
dict_out = {
    'system': [],
    'trial': [],
    'rep': [],
    't_on': [],
    't_peak': [],
    't_off100': [],
    't_off20': [],
    'v_peak': [],
    'eff_pos_mean': [],
    'eff_pos_std': []
    }