# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 16:39:23 2023

@author: zafar
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import label

# HELPER FUNCTIONS
def getOnsetOffset(x: np.array, threshold: float) -> (list, list):
    x = np.abs(x.copy())
    x[x<threshold] = 0
    
    lbl, num_lbl = label(x)
    onsets = []
    offsets = []
    for i in range(num_lbl):
        idx = np.where(lbl==(i+1))[0]
        event_time = idx[-1] - idx[0]
        if event_time > 50:
            onsets.append(idx[0])
            offsets.append(idx[-1])
    
    return onsets, offsets

# Read
trial_no = 1
df = pd.read_csv(f'./fitts_ot_{trial_no}.csv')
df = df.interpolate(method='cubic')

FS = 250 # Hertz
X = df.x.values # position: mm
T = df.time.values # time: sec
V = np.gradient(X) * FS # velocity: mm/s

# RELEVANT WINDOW
TIME_START = 1 # seconds
TIME_END = 8 # seconds
X = X[np.logical_and(T>TIME_START, T<TIME_END)]
V = V[np.logical_and(T>TIME_START, T<TIME_END)]
T = T[np.logical_and(T>TIME_START, T<TIME_END)]

#% Find thresholds
# Movement onset: >20mm/s
on_20, off_20 = getOnsetOffset(V, 20)
# Movement onset: >100mm/s
on_100, off_100 = getOnsetOffset(V, 100)
# Peak velocities
pk_vel, _ = find_peaks(np.abs(V), height=500, distance=20)
        
plt.plot(T, X)
plt.vlines(T[on_20], -125, 125, color='g')
plt.plot(T[on_20], X[on_20], 'go')
plt.plot(T[off_20], X[off_20], 'ro')
plt.plot(T[off_100], X[off_100], 'mo')
plt.plot(T[pk_vel], X[pk_vel], 'ko')