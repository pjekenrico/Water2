import netCDF4, os, matplotlib
from netCDF4 import Dataset
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from itertools import compress
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from visualization import TimeSeries, SateliteTimeSeries, geographic_plot
from clustering import sort_clusters, clustering, timestep_clustering, average_data, single_chemical_clustering
from global_land_mask import globe
from scipy.interpolate import griddata


def readSatData(path):
    '''
        Reads in NetCDF4 files from the given path and returns them as a numpy matrix.
        Outputs the longitude and latitude matrices.
    '''
    dataset = Dataset(path, mode='r')
    key = list(dataset.variables)[0]
    if key == 'Depth' or key == 'depth':
        key = list(dataset.variables)[1]

    print("\nReading in file at: " + path + "\nQuantity: " + key)

    time = dataset.variables['time']

    d = [dt.date(dd.year, dd.month, dd.day)
         for dd in netCDF4.num2date(time[:], time.units)]

    try:
        lons = dataset.variables['lon'][:]
        lats = dataset.variables['lat'][:]
    except:
        lons = dataset.variables['longitude'][:]
        lats = dataset.variables['latitude'][:]

    data = np.squeeze(dataset.variables[key][:])

    unit = dataset.variables[key].units

    dataset.close()

    print("Domain coordinates: " + str((np.min(lats), np.max(lats))) +
          ", " + str((np.min(lons), np.max(lons))))
    print("Domain dimensions (lat, lon): (" +
          str(lats.shape[0])+"," + str(lons.shape[0]) + ")")
    print("Time frame: " + str(d[0]) + " - " + str(d[-1]))
    print("Number of time steps: " + str(len(d)))

    return data, lons, lats, d, key, unit


def findClose(vector, reference, end='min', reverse = False):
    '''
    Search for the index at which vector has a similar value to 'reference'. When the 'min' is given, the closest
    value of vector is sought from below, returning an index of a value that is slightly smaller:
        vector = [1,2,3,4,5,6], reference = 2.5 -> i = 2

    Else, if the 'max' is used it returns the index to the next bigger entry in vector looking from above:
        vector = [1,2,3,4,5,6], reference = 2.5 -> i = 3

    vector: Iterable array or list of numbers.
    reference: Reference value.
    end: String stating whether the head of the vector is sought (lower match) or the tail.
    '''

    if not reverse:
        if end == 'min' or end == 'Min':
            for counter, value in enumerate(vector):
                if value > reference:
                    return max(counter-1, 0)

        elif end == 'max' or end == 'Max':
            for counter, value in reversed(list(enumerate(vector))):
                if value < reference:
                    return min(counter+1, len(vector)-1)
    else:
        if end == 'min' or end == 'Min':
            for counter, value in enumerate(vector):
                if value < reference:
                    return max(counter-1, 0)

        elif end == 'max' or end == 'Max':
            for counter, value in reversed(list(enumerate(vector))):
                if value > reference:
                    return min(counter+1, len(vector)-1)


def removeTimeSteps(data, rmIdx, dates):
    # Remove the data of unmatched time steps from the satelite data
    data1 = np.ma.copy(data)
    dates1 = dates.copy()
    for idxToRemove in reversed(rmIdx):
        dates1.pop(idxToRemove)
        data1 = data1[np.arange(len(data1[:, 0, 0])) != idxToRemove]

    return data1, dates1


class DataSet():

    def __init__(self, filename):
        [self.data, self.lons, self.lats, self.times, self.keys,
            self.unit] = readSatData(os.path.abspath(filename))


class SateliteData(DataSet):
    def __init__(self, filename):
        super().__init__(filename)

        if len(self.times) != 252:
            self.RefSet = DataSet('MetO-NWS-BIO-dm-CHL.nc')

            try:
                with np.load('region_labels.npz') as r_labels:
                    region_labels = r_labels['matrix']
            except:
                region_labels = region_calculation(
                    n_regions=4, show_silhouette=True)
                np.savez_compressed('region_labels.npz', matrix=region_labels)

            self.regionLabels = region_labels

            self.removeUnmatchingTime()

            self.removeEmptyLines()

            self.reduceSizeSpace()

            self.removeLandPixels()

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
        [self.data, self.times] = removeTimeSteps(
            self.data, removeSat, self.times)

        # Remove the data of unmatched time steps from the normal data
        [self.RefSet.data, self.RefSet.times] = removeTimeSteps(
            self.RefSet.data, removeNormal, self.RefSet.times)

        if self.times == self.RefSet.times and len(self.RefSet.data) == len(self.data):
            print("Successfully removed "+str(len(removeSat)) +
                  " elements from the satelite data.")
            print("Successfully removed "+str(len(removeNormal)) +
                  " elements from the normal data.\n")
        else:
            print("ERROR")

    def removeEmptyLines(self):
        alwaysFilled = np.ones((self.data[0].shape), dtype=bool)
        alwaysMasked = np.ones((self.data[0].shape), dtype=bool)

        rmCol = list()
        rmRow = list()

        for frame in self.data[:]:
            alwaysFilled = alwaysFilled & ~frame.mask
            alwaysMasked = alwaysMasked & frame.mask

        for i in range(len(self.data[0, :, 0])):
            if np.all(alwaysMasked[i, :]):
                rmCol.append(i)

        for i in range(len(self.data[0, 0, :])):
            if np.all(alwaysMasked[:, i]):
                rmRow.append(i)

        for idxToRemove in reversed(rmRow):
            self.data = self.data[:, np.arange(
                len(self.data[0, :, 0])) != idxToRemove, :]
            self.lons = np.delete(self.lons, idxToRemove)

        for idxToRemove in reversed(rmCol):
            self.data = self.data[:, np.arange(
                len(self.data[0, 0, :])) != idxToRemove, :]
            self.lats = np.delete(self.lats, idxToRemove)

        # If you want to see all the gaps along the time of the satelite data, just uncomment and plot
        # for i in range(len(self.data)):
        #    self.data[i,:,:].mask = ~alwaysFilled

    def reduceSizeSpace(self):
        print("\nRemoving excess space...")
        # Cut out excessive spacial data
        minLon = findClose(self.lons, self.RefSet.lons[0], end='min')
        maxLon = 270  # findClose(self.lons, self.RefSet.lons[-1], end = 'max')
        minLat = 35  # findClose(self.lats, self.RefSet.lats[0], end = 'min')
        maxLat = findClose(self.lats, self.RefSet.lats[-1], end='max')
        self.data = self.data[:, minLat: maxLat, minLon: maxLon]
        self.lons = self.lons[minLon: maxLon]
        self.lats = self.lats[minLat: maxLat]
        self.RefSet.data = self.RefSet.data[:, 1: 54, 10:100]
        self.RefSet.lats = self.RefSet.lats[1: 54]
        self.RefSet.lons = self.RefSet.lons[10: 100]
        self.regionLabels = self.regionLabels[1: 54, 10:100]
        print("Sat shape", np.shape(self.data))
        print("Model shape", np.shape(self.RefSet.data))


    def removeLandPixels(self):
        # Remove values that are on mainland or lakes

        print("Removing values on mainland and lakes...\n")

        lats, lons = np.meshgrid(self.lats, self.lons)

        isLand = globe.is_land(lats.T, lons.T)

        for k in range(len(self.data)):
            self.data[k].mask = isLand | self.data[k].mask

        latsRef, lonsRef = np.meshgrid(self.RefSet.lats, self.RefSet.lons)
        isLand = globe.is_land(latsRef.T, lonsRef.T)

        for k in range(len(self.RefSet.data)):
            self.RefSet.data[k].mask = isLand | self.RefSet.data[k].mask

        lats, lons = np.meshgrid(self.lons, self.lats)
        latsRef, lonsRef = np.meshgrid(self.RefSet.lons, self.RefSet.lats)

        newMask = griddata((lonsRef.flatten(), latsRef.flatten()), self.RefSet.data[0].mask.flatten(), (lons, lats), method='nearest')

        for k in range(len(self.data)):
            self.data[k].mask = newMask | self.data[k].mask

        print("Finished preprocessing.\n")


    def mapLabels(self, labels_model, lons_lats_model):
        '''
        Use nearest neighbour interpolation to map the given labels to the satellite set.

        labels_model :      Numpy array defined over the spatial coordinates of lons_lats_model.
        lons_lats_model :   Numpy array of spatial coordinates.
        '''

        print("\nMapping labels from the model data to the satelite coordinates...")

        lats, lons = np.meshgrid(self.lons, self.lats)
        latsRef = lons_lats_model[:,:,0].flatten()
        lonsRef = lons_lats_model[:,:,1].flatten()

        return griddata((lonsRef, latsRef), labels_model.flatten(), (lons, lats), method='nearest')

def clustervaluesSat(satData, satLabels, modelLabels, lon = 8.6865, lat = 54.025):
    '''
    Function to plot the yearly mean of two cuantities of a certain cluster. The preset values correspong
    to the river estuary of the Elba, Weser and Rhine rivers.

    matrix:     Matrix [timesteps, lons, lats] containing the unmodified model values.
    labels:     Labels of the clustering algorithm.
    lons_lats:  Spatial coordinates of the form [:,:,0:1].
    lon:        Longitude of a point within the cluster of interest.
    lat:        Latitude of a point within the cluster of interest.
    '''

    if not isinstance(satData, SateliteData):
                raise TypeError("Please provide SateliteData instance as input...")

    n = len(satData.times)

    # Compute stuff for satelite data
    lonEstuary = findClose(satData.lons, lon, end = 'min')
    latEstuary = findClose(satData.lats, lat, end = 'min', reverse = True)
    label_estuary_satelite = satLabels[latEstuary,lonEstuary]
    idxSat = np.where(satLabels == label_estuary_satelite)

    # Compute stuff for model data
    lonEstuary = findClose(satData.RefSet.lons, lon, end = 'min')
    latEstuary = findClose(satData.RefSet.lats, lat, end = 'min')
    label_estuary_model = modelLabels[latEstuary,lonEstuary]
    idxModel = np.where(modelLabels == label_estuary_satelite)

    satCHL = list()
    modelCHL = list()
    dates = [satData.times[0].year]
    datesIdx = [satData.times[0].year]
    years = np.array([date.year for date in satData.times])

    rawSatCHL = satData.data
    rawModelCHL = satData.RefSet.data
    meanSatCHL = np.zeros(n)
    meanModelCHL = np.zeros(n)

    for k in range(n):
        datesIdx.append(satData.times[k].year)

        if int(dates[-1]) < satData.times[k].year:
            dates.append(satData.times[k].year)

        tmp = rawSatCHL[k]
        meanSatCHL[k] = np.nanmean(tmp[idxSat])
        tmp = rawModelCHL[k]
        meanModelCHL[k] = np.nanmean(tmp[idxModel])
    
    datesIdx.pop(-1)
    datesIdx = np.array(datesIdx)

    for k in dates:
        satCHL.append(np.mean(meanSatCHL[np.where(datesIdx == k)]))
        modelCHL.append(np.mean(meanModelCHL[np.where(datesIdx == k)]))

    satCHL = np.array(satCHL)
    modelCHL = np.array(modelCHL)

    fig, ax = plt.subplots(figsize = (8,6))

    color = 'tab:red'
    ax.set_xlabel('Year', fontdict=dict(size=12))
    ax.set_ylabel(r'$Chl~[\frac{mg}{m^3}]$', fontdict=dict(size=14))
    ax.plot(dates, satCHL, color=color, label = 'Satelite')
    ax.set_xticks(dates[0::2])
    ax.set_xticklabels(dates[0::2])
    ax.set_xlim([np.min(dates[0::2]), np.max(dates[0::2])])

    color = 'tab:blue'
    ax.plot(dates, modelCHL, color = color, label = 'Model')
    ax.set_title("Mean annual concentrations of chlorophyll", fontdict=dict(color="black", size=14))
    ax.grid('on', axis = 'x', linewidth = 0.5, linestyle = '--', alpha = 0.5)

    plt.legend()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()




def __main__():
    chl_path = 'dataset-CHL-satellite-daily.nc'
    spm_path = 'dataset-SPM-satellite-monthly.nc'

    sat1 = SateliteData(chl_path)

    #with np.load('lons_lats.npz') as ll:
    #    lons_lats = ll['lons_lats']

    #try:
    #    with np.load('region_labels.npz') as r_labels:
    #        region_labels = r_labels['matrix']
    #except:
    #    region_labels = region_calculation(
    #        n_regions=4, show_silhouette=True)
    #    np.savez_compressed('region_labels.npz', matrix=region_labels)

    #satLabels = sat1.mapLabels(region_labels,lons_lats)

    #clustervaluesSat(sat1, satLabels, sat1.regionLabels, lon = 8.6865, lat = 54.025)

    satData = np.asarray(sat1.data)
    satData = np.where(sat1.data.mask, np.nan, satData)

    modelData = np.asarray(sat1.RefSet.data)
    modelData = np.where(sat1.RefSet.data.mask, np.nan, modelData)

    for i in range(satData.shape[0]):
        if i < satData.shape[0] - 1 and i > 0:
            idxWhereNan = np.where(np.isnan(satData[i, :, :]) & (~np.isnan(satData[i-1, :, :]) | ~np.isnan(satData[i+1,:,:])))
            if not idxWhereNan[0].size == 0:
                tmp = satData[i-1, :, :]
                tmp2 = satData[i, :, :]
                tmp3 = satData[i+1, :, :]
                tmp2[idxWhereNan] = np.nanmean([tmp[idxWhereNan], tmp2[idxWhereNan], tmp3[idxWhereNan]], axis = 0)
                satData[i, :, :] = tmp2

        a = satData[i,:,:]
        satData[i,:,:]= (a-np.full(a.shape,np.amin(a[~np.isnan(a)])))/(np.amax(a[~np.isnan(a)])-np.amin(a[~np.isnan(a)]))
        b = modelData[i,:,:]
        modelData[i,:,:]= (b-np.full(b.shape,np.amin(b[~np.isnan(b)])))/(np.amax(b[~np.isnan(b)])-np.amin(b[~np.isnan(b)]))

    from clustering import single_chemical_clustering
    [cl_data_sat, labels_sat, cl_sizes_sat, s_avg] = single_chemical_clustering(matrix = satData,\
       n_clusters=3, verbose=True)
    [cl_data_model, labels_model, cl_sizes_model, s_avg] = single_chemical_clustering(matrix = modelData,\
       n_clusters=3, verbose=True)

    lons, lats = np.meshgrid(sat1.lons, sat1.lats)
    lons_lats = np.zeros((lons.shape[0],lons.shape[1],2))
    lons_lats[:,:,0] = lons
    lons_lats[:,:,1] = lats

    geographic_plot(labels_sat, lons_lats, cluster = True,\
        adjustBorder = False, levels = 2, title = r'Single chemical clustering - CHL (Satellite)')

    lons, lats = np.meshgrid(sat1.RefSet.lons, sat1.RefSet.lats)
    lons_lats = np.zeros((lons.shape[0],lons.shape[1],2))
    lons_lats[:,:,0] = lons
    lons_lats[:,:,1] = lats

    geographic_plot(labels_model, lons_lats, cluster = True,\
        adjustBorder = True, levels = 2, title = r'Single chemical clustering - CHL (Model)')

    print('Done')

if __name__ == "__main__":
    __main__()
