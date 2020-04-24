import netCDF4
from netCDF4 import Dataset
import os
import numpy as np
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
from itertools import compress
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def readSatData(path):
    '''
        Reads in NetCDF4 files from the given path and returns them as a numpy matrix.
        Outputs the longitude and latitude matrices.
    '''
    dataset = Dataset(path, mode = 'r')
    key = list(dataset.variables)[0]
    print("\nReading in file at: " + path + "\nQuantity: " + key)

    time = dataset.variables['time']

    d = [dt.date(dd.year, dd.month, dd.day) for dd in netCDF4.num2date(time[:], time.units)]

    try:
        lons = dataset.variables['lon'][:]
        lats = dataset.variables['lat'][:]
    except:
        lons = dataset.variables['longitude'][:]
        lats = dataset.variables['latitude'][:]

    data = np.squeeze(dataset.variables[key][:])

    dataset.close()

    print("Domain coordinates: " + str((np.min(lats), np.max(lats))) +", "+ str((np.min(lons), np.max(lons))))
    print("Domain dimensions (lat, lon): " + str(lons.shape))
    print("Time frame: " + str(d[0]) + " - " +str(d[-1]))
    print("Number of time steps: " + str(len(d)))

    return data, lons, lats, d




chl_path = os.path.abspath('dataset-CHL-satellite-daily.nc')
spm_path = os.path.abspath('dataset-SPM-satellite-monthly.nc')

[data, lons, lats, times] = readSatData(spm_path)




lons, lats = np.meshgrid(lons, lats)
timestep = 100

# Plotting the clusters
matplotlib.rcParams['figure.figsize'] = (10, 10)
proj = ccrs.Mercator()
m = plt.axes(projection=proj)

# Put a background image on for nice sea rendering.
m.stock_img()
m.coastlines(resolution='10m')
m.add_feature(cfeature.BORDERS)
gl = m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False

# Plot data
plt.contourf(lons, lats, data[timestep,:,:], 50, transform=ccrs.PlateCarree())

# Add Colorbar
cbar = plt.colorbar()

# Add Title
plt.title('Time: ' + str(times[timestep]) + ' clusters')

plt.show()

# np.save('sat_model_data.npy', matrix)
#np.save('lons_lats.npy', lons_lats)
