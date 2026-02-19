#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 01:11:59 2025

@author: juliusfueg

This script tries to tackle the problem of HRTF generation. The idea of the
present approach is to use the K-Nearest-Neighbor-Algorithm in order to
learn a set of HRTFs with different azimuth and elevation angles.
This learned information is then used in order to generate HRTFs of 
new angular positions by interpolation. 
"""

import pysofaconventions as sofa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

### read in sofa file ###
hrtf = sofa.SOFAFile('SADIE_003_DFC_256_order_fir_48000.sofa', 'r')

# source positions: azimuth, elevation, radius
positions = hrtf.getVariableValue('SourcePosition') 

# sampling rate
fs = hrtf.getSamplingRate()

# get IR data
data_ir = hrtf.getDataIR()

### plot IR positions ###
# get azimuth and elevation angles in degrees
azimuth_deg = positions[:, 0]
elevation_deg = positions[:, 1]

# norm distance to 1, so that all angles are on a unity sphere
r = 1

# convert degrees to radians
azimuth = np.radians(azimuth_deg)
elevation = np.radians(elevation_deg)

# convert spherical to cartesian coordinates (unity sphere)
x = r * np.cos(elevation) * np.cos(azimuth)
y = r * np.cos(elevation) * np.sin(azimuth)
z = r * np.sin(elevation)

# plot all present IRs on a sphere
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

# draw sphere
u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
xs = np.cos(u)*np.sin(v)
ys = np.sin(u)*np.sin(v)
zs = np.cos(v)
ax.plot_surface(xs, ys, zs, color='c', alpha=0.1, edgecolor='none')

# plot IR positions
ax.scatter(x, y, z, color='r')

# set axis limits
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# labels of axes
ax.set_xlabel('X (Vorne + / Hinten -)')
ax.set_ylabel('Y (Rechts + / Links -)')
ax.set_zlabel('Z (Oben + / Unten -)')

# add directions for orientation
ax.set_xticks([-1, 0, 1])
ax.set_xticklabels(['Hinten', '', 'Vorne'])

ax.set_yticks([-1, 0, 1])
ax.set_yticklabels(['Links', '', 'Rechts'])

ax.set_zticks([-1, 0, 1])
ax.set_zticklabels(['Unten', '', 'Oben'])

plt.show()

### knn ###
# in this example, a set of 170 HRTFs and their coordinates (azimuth, 
# elevation, distance) is used

# read in IR positions in degrees
azimuth_deg = positions[:,0]
elevation_deg = positions[:,1]
r = 1

# convert degrees to radians
azimuth = np.radians(azimuth_deg)
elevation = np.radians(elevation_deg)

# convert spherical to cartesian coordinates (unity sphere)
X = np.zeros((len(positions), 3))
X[:,0] = r * np.cos(elevation) * np.cos(azimuth)
X[:,1] = r * np.cos(elevation) * np.sin(azimuth)
X[:,2] = r * np.sin(elevation)

# take the mean of the two HRTFs of a position for simplicity of the example
hrtfs_mono = data_ir.mean(axis=1)
y = hrtfs_mono # rename for simplicity

# 1. split into train/test data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. scale (important for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. initialize KNN regressor and train
knn = KNeighborsRegressor(n_neighbors=7, metric='manhattan', weights='distance')  # n_neighbors can be adjusted
knn.fit(X_train_scaled, y_train)

# 4. prediction of test data
y_pred = knn.predict(X_test_scaled)

# 5. evaluation (MSE per data point, dann Mittelwert)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error of test data: {mse:.6f}")

# regression accuracy in percentage
variance = np.var(y_test)
accuracy = 100 * (1 - mse / variance)
print(f"precision of approximation (based on variance): {accuracy:.2f}%")

# 6. accuracy directly from the model
print("accuracy (over model.score):", knn.score(X_test, y_test))

### plot example of approximated HRTF ###
# example: compare original and generated HRTFs
index = 0  # or np.random.randint(0, len(y_test))

true_hrtf = y_test[index]
pred_hrtf = y_pred[index]

plt.figure(figsize=(10, 5))
plt.plot(true_hrtf, label='Original HRTF', linewidth=2)
plt.plot(pred_hrtf, label='Predicted HRTF', linestyle='--')
plt.title('Comparison: Original vs. Predicted HRTF')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

def compute_snr(true_hrtf, pred_hrtf):
    signal_power = np.sum(true_hrtf ** 2)
    noise_power = np.sum((true_hrtf - pred_hrtf) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# SNR of example
snr_value = compute_snr(true_hrtf, pred_hrtf)
print(f"SNR: {snr_value:.2f} dB")
