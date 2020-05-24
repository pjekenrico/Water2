import netCDF4
from netCDF4 import Dataset
import os
import numpy as np
from scipy import stats
import datetime as dt
import sys
import pickle

datasets = []


# Modes of rescaling the data:
# 0to1_dayly:   scales the data linearly from 0 to 1; 0 is daily min and 1 is daily max
# zscores:      scales the data based on daily zscores
mode = '0to1_dayly'

# Loading data from datasets
path_to_files = os.path.abspath('')+'\MetO-NWS-BIO-dm-'
extension = ".nc"

chl_path = path_to_files + "CHL" + extension
datasets.append(Dataset(chl_path, mode='r'))
doxy_path = path_to_files + "DOXY" + extension
datasets.append(Dataset(doxy_path, mode='r'))
nitr_path = path_to_files + "NITR" + extension
datasets.append(Dataset(nitr_path, mode='r'))
phos_path = path_to_files + "PHOS" + extension
datasets.append(Dataset(phos_path, mode='r'))

# Printing the keys (not relevant)
keys = []
keys.append(list(datasets[0].variables)[1])
keys.append(list(datasets[1].variables)[0])
keys.append(list(datasets[2].variables)[0])
keys.append(list(datasets[3].variables)[1])

print(keys)


# Transforming tha dates into datetime dates (not saved)
time = datasets[0].variables['time']
jd = netCDF4.num2date(time[:], time.units)
d = []
for dd in jd:
    d.append(dt.date(dd.year, dd.month, dd.day))

# Loading the langitudes and latitudes data
lons = datasets[0].variables['longitude'][:]
lats = datasets[0].variables['latitude'][:]

lons, lats = np.meshgrid(lons, lats)

# Storing relevant data in a different List
data = []
for i in range(4):
    data.append(np.squeeze(datasets[i].variables[keys[i]][:]))

# Closing opened datasets
for i in range(4):
    datasets[i].close()

# Transforming the data in a np.array
matrix = np.zeros((data[0].shape[0], data[0].shape[1], data[0].shape[2], 4))
matrix[:, :, :, 0] = np.asarray(data[0])
matrix[:, :, :, 1] = np.asarray(data[1])
matrix[:, :, :, 2] = np.asarray(data[2])
matrix[:, :, :, 3] = np.asarray(data[3])

matrix = np.where(matrix < 0, np.nan, matrix)

# Performing standardization
if mode == 'zscore':
    for i in range(matrix.shape[0]):
        a = matrix[i, :, :, 0]
        matrix[i, :, :, 0] = (
            a - np.full(a.shape, np.mean(a[~np.isnan(a)])))/np.std(a[~np.isnan(a)])
        a = matrix[i, :, :, 1]
        matrix[i, :, :, 1] = (
            a - np.full(a.shape, np.mean(a[~np.isnan(a)])))/np.std(a[~np.isnan(a)])
        a = matrix[i, :, :, 2]
        matrix[i, :, :, 2] = (
            a - np.full(a.shape, np.mean(a[~np.isnan(a)])))/np.std(a[~np.isnan(a)])
        a = matrix[i, :, :, 3]
        matrix[i, :, :, 3] = (
            a - np.full(a.shape, np.mean(a[~np.isnan(a)])))/np.std(a[~np.isnan(a)])
elif mode == '0to1_dayly':
    for i in range(matrix.shape[0]):
        a = matrix[i, :, :, 0]
        matrix[i, :, :, 0] = (
            a - np.full(a.shape, np.amin(a[~np.isnan(a)])))/(np.amax(a[~np.isnan(a)]) - np.amin(a[~np.isnan(a)]))
        a = matrix[i, :, :, 1]
        matrix[i, :, :, 1] = (
            a - np.full(a.shape, np.amin(a[~np.isnan(a)])))/(np.amax(a[~np.isnan(a)]) - np.amin(a[~np.isnan(a)]))
        a = matrix[i, :, :, 2]
        matrix[i, :, :, 2] = (
            a - np.full(a.shape, np.amin(a[~np.isnan(a)])))/(np.amax(a[~np.isnan(a)]) - np.amin(a[~np.isnan(a)]))
        a = matrix[i, :, :, 3]
        matrix[i, :, :, 3] = (
            a - np.full(a.shape, np.amin(a[~np.isnan(a)])))/(np.amax(a[~np.isnan(a)]) - np.amin(a[~np.isnan(a)]))

# Transforming lat and lon data in a np.array
lons_lats = np.zeros((lons.shape[0], lons.shape[1], 2))
lons_lats[:, :, 0] = np.asarray(lons)
lons_lats[:, :, 1] = np.asarray(lats)

# Saving the data
np.savez_compressed('model_data.npz', matrix=matrix)
np.savez_compressed('lons_lats.npz', lons_lats=lons_lats)

# Saving dates using pickle
with open("datetimes.txt", "wb") as fp:
    pickle.dump(d, fp)
