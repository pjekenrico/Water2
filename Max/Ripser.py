#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:24:42 2020

@author: max
"""
import netCDF4
from netCDF4 import Dataset
import os
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
from math import pi
import math
from numpy import cos,sin
from scipy.spatial import cKDTree
from ripser import ripser, Rips
import persim as ps
import kmapper
import matplotlib.pyplot as plt
import sys
import copy
from sklearn.metrics.pairwise import pairwise_distances
from skimage.util.shape import view_as_blocks
import gudhi
from mpl_toolkits.mplot3d import Axes3D

#def dist(a, b):
#    d = [a[0] - b[0], a[1] - b[1]]
#    return np.sqrt(d[0] * d[0] + d[1] * d[1])

def Grouper(A,D,eps, group1):
    ITEMS = set(range(len(A)))

    for I in range(len(A)):
        group1_temp = set()
        for J in ITEMS - group1:
            M = min(min(D[J,j],D[j,J]) for j in group1)
            if M<=eps:
                group1_temp = group1_temp.union({J})
        if len(group1_temp) == 0:
            break
        else:
            group1 = group1.union(group1_temp)

    group1 = list(group1)
    return A[group1,:]



nc_path = os.path.abspath("dataset-DOXYL-model-daily.nc"); 
dataset = Dataset(nc_path)

attr=dataset.ncattrs() #find all NetCDF global attributes
#==============================================================================  
timestep=400; #choose timestep
lon_lat_dim = 2

fh = Dataset(nc_path, mode='r')
time=fh.variables['time']
jd = netCDF4.num2date(time[:],time.units)
lons = fh.variables['longitude'][:]
lats = fh.variables['latitude'][:]

lons, lats = np.meshgrid(lons,lats)

Chlfa = fh.variables['o2'][:]
Chlfa = np.squeeze(Chlfa)
fh.close()

Chlfa_plot=Chlfa[timestep,:,:]
#lst = []
#
#avg = view_as_blocks(Chlfa_plot,block_shape=(9,9))
#sys.exit()
#for i in range(0,125):
#    for j in range(0,125):
#        average = np.mean(avg[i,j])
#        lst.append(average)
#
#new = np.array([lst])
#
#new = new.reshape((125,125))

newarray = np.array([[None,None,None]], dtype = np.float32)
for i in range(len(Chlfa_plot)):
    for j in range(len(Chlfa_plot[0])):
        if math.isnan(Chlfa_plot[i][j]):
            continue
        else:
            newarray = np.append(newarray, np.array([[i,j, round(Chlfa_plot[i][j]/10)*10 ]]), axis = 0)

newarray = newarray[1:]
dictionary = ripser(newarray)
diagrams = dictionary['dgms']
distance = dictionary['dperm2all']

dist = 10
A = newarray
B = Grouper(A, distance, dist, {4235})
C = Grouper(A, distance, dist, {2012})
D = Grouper(A, distance, dist, {3808})
E = Grouper(A, distance, dist, {2})
F = Grouper(A, distance, dist, {31})
G = Grouper(A, distance, dist, {1439})
H = Grouper(A, distance, dist, {1145})

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(B[:,0], B[:,1], B[:,2], c = 'black', label = 'cluster')
ax.scatter(C[:,0], C[:,1], C[:,2], c = 'red', label = 'cluster')
ax.scatter(D[:,0], D[:,1], D[:,2], c = 'blue', label = 'cluster')
ax.scatter(E[:,0], E[:,1], E[:,2], c = 'yellow', label = 'cluster')
ax.scatter(F[:,0], F[:,1], F[:,2], c = 'cyan', label = 'cluster')
ax.scatter(G[:,0], G[:,1], G[:,2], c = 'magenta', label = 'cluster')
ax.scatter(H[:,0], H[:,1], H[:,2], c = '#00ffe5', label = 'cluster')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Value')
ax.scatter(A[:,0], A[:,1], A[:,2], c = 'green', s = 0.2, label = 'Original data')
ax.set_xlim(78,0)

sys.exit()

dictionary = ripser(newarray)
diagrams = dictionary['dgms']
distance = dictionary['dperm2all']

