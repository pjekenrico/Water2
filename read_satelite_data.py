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
from visualization import TimeSeries, SateliteTimeSeries, geographic_plot


def readSatData(path):
    '''
        Reads in NetCDF4 files from the given path and returns them as a numpy matrix.
        Outputs the longitude and latitude matrices.
    '''
    dataset = Dataset(path, mode = 'r')
    key = list(dataset.variables)[0]
    if key == 'Depth' or key == 'depth':
        key = list(dataset.variables)[1]

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
    
    unit = dataset.variables[key].units

    dataset.close()

    print("Domain coordinates: " + str((np.min(lats), np.max(lats))) +", "+ str((np.min(lons), np.max(lons))))
    print("Domain dimensions (lat, lon): (" + str(lats.shape[0])+","+ str(lons.shape[0]) +")")
    print("Time frame: " + str(d[0]) + " - " +str(d[-1]))
    print("Number of time steps: " + str(len(d)))

    return data, lons, lats, d, key, unit

def findClose(vector, reference, end = 'min'):
    '''
    Search for the index at which vector has a similar value to 'reference'. When the 'min' is given, the closest
    value of vector is sought from below, returning an index of a value that is slightly smaller:
        vector = [1,2,3,4,5,6], reference = 2.5 -> i = 2
    Else, if the 'max' is used it returns the index to the next bigger entry in vector looking from above:
        vector = [1,2,3,4,5,6], reference = 2.5 -> i = 3

    param vector: Iterable array or list of numbers.
    param reference: Reference value.
    param end: String stating whether the head of the vector is sought (lower match) or the tail.
    '''
    if end == 'min' or end == 'Min':
        for counter, value in enumerate(vector):
            if value > reference:
                return max(counter-1,0)

    elif end == 'max' or end == 'Max':
        for counter, value in reversed(list(enumerate(vector))):
            if value < reference:
                return min(counter+1, len(vector)-1)

def removeTimeSteps(data, rmIdx, dates):
    # Remove the data of unmatched time steps from the satelite data
    data1 = np.ma.copy(data)
    dates1 = dates.copy()
    for idxToRemove in reversed(rmIdx):
        dates1.pop(idxToRemove)
        data1 = data1[np.arange(len(data1[:,0,0]))!=idxToRemove]

    return data1, dates1

class DataSet():

    def __init__(self, filename):
        [self.data, self.lons, self.lats, self.times, self.keys, self.unit] = readSatData(os.path.abspath(filename))

class SateliteData(DataSet):
    def __init__(self, filename):
        super().__init__(filename)
        self.RefSet = DataSet('dataset-CHL-model-daily.nc')

        self.removeUnmatchingTime()

        self.removeEmptyLines()

        self.reduceSizeSpace()

    def removeUnmatchingTime(self):

        print("\nRemoving non overlapping days from lists...")
        removeSat = list()
        removeNormal = list()

        # Look for dates that have to be removed from the satelite data
        for counter, elem in enumerate(self.times):
            if elem not in self.RefSet.times:
                removeSat.append(counter)

        # Look for dates that have to be removed from the normal data
        for counter, elem in enumerate(self.RefSet.times):
            if elem not in self.times:
                removeNormal.append(counter)

        # Remove the data of unmatched time steps from the satelite data
        [self.data, self.times] = removeTimeSteps(self.data, removeSat, self.times)

        # Remove the data of unmatched time steps from the normal data
        [self.RefSet.data, self.RefSet.times] = removeTimeSteps(self.RefSet.data, removeNormal, self.RefSet.times)

        if self.times == self.RefSet.times and len(self.RefSet.data) == len(self.data):
            print("Successfully removed "+str(len(removeSat))+" elements from the satelite data.")
            print("Successfully removed "+str(len(removeNormal))+" elements from the normal data.\n")
        else:
            print("ERROR")

    def removeEmptyLines(self):
        alwaysFilled = np.ones((self.data[0].shape), dtype = bool)
        alwaysMasked = np.ones((self.data[0].shape), dtype = bool)

        rmCol = list()
        rmRow = list()

        for frame in self.data[:]:
            alwaysFilled = alwaysFilled & ~frame.mask
            alwaysMasked = alwaysMasked &  frame.mask

        for i in range(len(self.data[0,:,0])):
            if np.all(alwaysMasked[i,:]):
                rmCol.append(i)

        for i in range(len(self.data[0,0,:])):
            if np.all(alwaysMasked[:,i]):
                rmRow.append(i)

        for idxToRemove in reversed(rmRow):
            self.data = self.data[:, np.arange(len(self.data[0,:,0])) != idxToRemove,:]
            self.lons = np.delete(self.lons, idxToRemove)

        for idxToRemove in reversed(rmCol):
            self.data = self.data[:, np.arange(len(self.data[0,0,:])) != idxToRemove,:]
            self.lats = np.delete(self.lats, idxToRemove)

        # If you want to see all the gaps along the time of the satelite data, just uncomment and plot
        #for i in range(len(self.data)):
        #    self.data[i,:,:].mask = ~alwaysFilled

    def reduceSizeSpace(self):
        print("\nRemoving excess space...")
        # Cut out excessive spacial data
        minLon = findClose(self.lons, self.RefSet.lons[0], end = 'min')
        maxLon = 270 #findClose(self.lons, self.RefSet.lons[-1], end = 'max')
        minLat = 35 #findClose(self.lats, self.RefSet.lats[0], end = 'min')
        maxLat = findClose(self.lats, self.RefSet.lats[-1], end = 'max')
        self.data = self.data[:, minLat : maxLat, minLon : maxLon]
        self.lons = self.lons[minLon : maxLon]
        self.lats = self.lats[minLat : maxLat]
        self.RefSet.data = self.RefSet.data[:, 1 : 54, 10:100]
        self.RefSet.lats = self.RefSet.lats[1 : 54]
        self.RefSet.lons = self.RefSet.lons[10 : 100]


def __main__():
    chl_path = 'dataset-CHL-satellite-daily.nc'
    spm_path = 'dataset-SPM-satellite-monthly.nc'

    sat1 = SateliteData(chl_path)
    #sat2 = SateliteData('dataset-SPM-satellite-monthly.nc')

    
    myAnimation = SateliteTimeSeries(sat1)

    max_data_value = [10, 10]
    min_data_value = [0, 0]

    # Create animation
    myAnimation.createAnimation(number_of_contour_levels = 20, n_rows = 1, n_cols = 2,\
        max_data_value = max_data_value, min_data_value = min_data_value, start_frame = 100,\
        end_frame = 100, skip_frames = 100)

    # myAnimation.saveAnimation(fps = 8, name = 'toLazytoName2')

    # lons, lats = np.meshgrid(sat1.lons, sat1.lats)
    # lons_lats = np.zeros((lons.shape[0],lons.shape[1],2))
    # lons_lats[:,:,0] = lons
    # lons_lats[:,:,1] = lats

    # timestep = 4000

    # geographic_plot(sat1.data[timestep,:,:], lons_lats, key = sat1.keys+' (Sat)',\
    #   unit = sat1.unit, date = sat1.times[timestep], minVal = None,\
    #   maxVal = 0.5*np.nanmax(sat1.data[timestep,:,:]), adjustBorder = False)

    timestep = 5000

    lons, lats = np.meshgrid(sat1.RefSet.lons, sat1.RefSet.lats)
    lons_lats = np.zeros((lons.shape[0],lons.shape[1],2))
    lons_lats[:,:,0] = lons
    lons_lats[:,:,1] = lats

    # geographic_plot(sat1.RefSet.data[timestep,:,:], lons_lats, key = sat1.RefSet.keys,\
    #    unit = sat1.RefSet.unit, date = sat1.RefSet.times[timestep], minVal = None,\
    #    maxVal = 0.8*np.nanmax(sat1.RefSet.data[timestep,:,:]), adjustBorder = False)

if __name__ == "__main__":
    __main__()