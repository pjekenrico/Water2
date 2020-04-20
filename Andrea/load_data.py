import netCDF4
from netCDF4 import Dataset
import os
import numpy as np
import datetime as dt

datasets = []

chl_path = "C:/Users/andre/OneDrive/Desktop/TUD/Mathematical Data Science/Project/Water2/Andrea/MetO-NWS-BIO-dm-CHL.nc"
datasets.append(Dataset(chl_path, mode='r'))
doxy_path = "C:/Users/andre/OneDrive/Desktop/TUD/Mathematical Data Science/Project/Water2/Andrea/MetO-NWS-BIO-dm-DOXY.nc"
datasets.append(Dataset(doxy_path, mode='r'))
nitr_path = "C:/Users/andre/OneDrive/Desktop/TUD/Mathematical Data Science/Project/Water2/Andrea/MetO-NWS-BIO-dm-NITR.nc"
datasets.append(Dataset(nitr_path, mode='r'))
phos_path = "C:/Users/andre/OneDrive/Desktop/TUD/Mathematical Data Science/Project/Water2/Andrea/MetO-NWS-BIO-dm-PHOS.nc"
datasets.append(Dataset(phos_path, mode='r'))

keys = []
keys.append(list(datasets[0].variables)[1])
keys.append(list(datasets[1].variables)[0])
keys.append(list(datasets[2].variables)[0])
keys.append(list(datasets[3].variables)[1])

print(keys)

time = datasets[0].variables['time']
jd = netCDF4.num2date(time[:], time.units)
d = []
for dd in jd:
    d.append(dt.date(dd.year, dd.month, dd.day))

lons = datasets[0].variables['longitude'][:]
lats = datasets[0].variables['latitude'][:]

lons, lats = np.meshgrid(lons, lats)

data = []
for i in range(4):
    data.append(np.squeeze(datasets[i].variables[keys[i]][:]))

for i in range(4):
    datasets[i].close()

matrix = np.zeros((data[0].shape[0], data[0].shape[1], data[0].shape[2], 4))
matrix[:, :, :, 0] = data[0]
matrix[:, :, :, 1] = np.asarray(data[1])
matrix[:, :, :, 2] = np.asarray(data[2])
matrix[:, :, :, 3] = np.asarray(data[3])

lons_lats = np.zeros((lons.shape[0], lons.shape[1], 2))
lons_lats[:, :, 0] = np.asarray(lons)
lons_lats[:, :, 1] = np.asarray(lats)

np.save('model_data.npy', matrix)
np.save('lons_lats.npy', lons_lats)
