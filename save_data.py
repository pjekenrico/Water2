import netCDF4
from netCDF4 import Dataset
import os
import numpy as np
from scipy import stats
import datetime as dt
import sys
import pickle

def average_data(matrix=None, chemicals=[True, True, True, True], delta_t=10):
    '''
    This function averages the data through time

    matrix:     data through time, space and chemicals
    chemicals:  boolean array that represent which chemicals are being averaged;
                as default, all of them are averaged
    delta_t:    time period over which to average the data
    '''

    n_chemicals = sum(np.multiply(chemicals, 1))
    time_steps = int(matrix.shape[0]/delta_t)
    data = np.full(
        (time_steps, matrix.shape[1], matrix.shape[2], n_chemicals), np.nan)

    print("Starting Averaging Procedure:")
    sys.stdout.write("[%s]" % (" " * 10))
    sys.stdout.flush()
    sys.stdout.write("\b" * (10+1))
    count = 0.1
    for t in range(time_steps):
        layer = np.zeros((matrix.shape[1], matrix.shape[2], n_chemicals))
        for i in range(matrix.shape[1]):
            for j in range(matrix.shape[2]):
                t1 = t*delta_t
                t2 = min((t + 1)*delta_t, matrix.shape[0] - 1)
                c = 0
                for chem in range(len(chemicals)):
                    if chemicals[chem]:
                        temp_data = matrix[t1:t2, i, j, chem]
                        not_nan_indexes = ~np.isnan(temp_data)
                        n_not_nans = sum(np.multiply(not_nan_indexes, 1))
                        if n_not_nans > 0:
                            d = np.sum(temp_data[not_nan_indexes])/n_not_nans
                            layer[i, j, c] = d
                        else:
                            layer[i, j, c] = np.nan
                        c += 1
        data[t, :, :, :] = layer
        if t / time_steps >= count:
            sys.stdout.write(chr(9608))
            sys.stdout.flush()
            count += 0.1

    sys.stdout.write(chr(9608))
    sys.stdout.flush()
    print("\nFinished Averaging.")

    return data[:, :, :, :]


datasets = []

# 0to1_dayly: scales the data linearly from 0 to 1; 0 is daily min and 1 is daily max
# zscores: scales the data based on daily zscores 
mode = '0to1_dayly'

# Loading data from datasets
path_to_files = os.path.abspath('')+'\MetO-NWS-BIO-dm-'
#path_to_files = "C:/Users/andre/OneDrive/Desktop/TUD/Mathematical Data Science/Project/Water2/Andrea/MetO-NWS-BIO-dm-"
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
            a - np.full(a.shape, np.amin(a[~np.isnan(a)])))/(np.amax(a[~np.isnan(a)]) -  np.amin(a[~np.isnan(a)]))
        a = matrix[i, :, :, 1]
        matrix[i, :, :, 1] = (
            a - np.full(a.shape, np.amin(a[~np.isnan(a)])))/(np.amax(a[~np.isnan(a)]) -  np.amin(a[~np.isnan(a)]))
        a = matrix[i, :, :, 2]
        matrix[i, :, :, 2] = (
            a - np.full(a.shape, np.amin(a[~np.isnan(a)])))/(np.amax(a[~np.isnan(a)]) -  np.amin(a[~np.isnan(a)]))
        a = matrix[i, :, :, 3]
        matrix[i, :, :, 3] = (
            a - np.full(a.shape, np.amin(a[~np.isnan(a)])))/(np.amax(a[~np.isnan(a)]) -  np.amin(a[~np.isnan(a)]))

# Transforming lat and lon data in a np.array
lons_lats = np.zeros((lons.shape[0], lons.shape[1], 2))
lons_lats[:, :, 0] = np.asarray(lons)
lons_lats[:, :, 1] = np.asarray(lats)

# Saving the data
np.savez_compressed('model_data.npz', matrix=matrix)
np.savez_compressed('lons_lats.npz', lons_lats=lons_lats)
av_matrix = average_data(matrix=matrix, delta_t=100)
np.savez_compressed('av_model_data100.npz', matrix=av_matrix)

with open("datetimes.txt", "wb") as fp:   #Pickling
    pickle.dump(d, fp)

