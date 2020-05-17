# -*- coding: utf-8 -*-
import os
import netCDF4
import string
from netCDF4 import Dataset
import numpy as np
import datetime as dt
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import cm


def timeseries_plot(data=None, t=None):
    '''
    data:   labels
    t:      dates
    '''
    for i in range(int(max(data))+1):
        labels = []
        time = []
        for j in range(len(t)):
            if data[j] == i:
                labels.append(i)
                time.append(t[j])
        plt.plot_date(time, labels, markersize=4)
    # plt.plot_date(t, data)
    plt.show()


def geographic_plot(data, lons_lats=None, levels=5, key=None, unit=None, date=None, minVal=None, maxVal=None, adjustBorder=True):
    '''
        Plot single data frames.

        param data: 2D numpy array
        param lons_lats: Array of the shape [:,:,2] containing the longitudes in the first layer and the
        latitudes on the second layer.
        param key: String that denotes the quantity to be displayed e.g. "$NO_3$"
        param unit: (String) Unit of the displayed quantity e.g. "$mmol/m^3$"
        param date: (datetime) Date in datetime format (will be casted to string...)
        param minVal: (float) Lower bound for the value range. Lower values are capped at minVal.
        param maxVal: (float) Uower bound for the value range. Larger values are capped at maxVal.
    '''

    # Plotting the clusters
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.Mercator())

    # Put a background image on for nice sea rendering.
    # ax.stock_img()

    # High resolution map features
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land', '10m'),
                   linestyle=':', facecolor='none', edgecolor='black')
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '10m'),
                   facecolor='none', edgecolor='blue')

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabels_top = False
    gl.ylabels_right = False

    # Adapt value ranges according to minVal and maxVal
    if maxVal is None and not minVal is None:
        data = data*(data >= minVal) + minVal*(data <= minVal)
    elif minVal is None and not maxVal is None:
        data = data*(data <= maxVal) + maxVal*(data >= maxVal)
    elif not maxVal is None and not minVal is None:
        data = data*(data <= maxVal)*(data >= minVal) + minVal * \
            (data <= minVal) + maxVal*(data >= maxVal)

    data[0, 0] = minVal
    data[0, 1] = maxVal

    # Plot data
    cs = plt.contourf(lons_lats[:, :, 0], lons_lats[:, :, 1], data, levels,
                      cmap=cm.rainbow, transform=ccrs.PlateCarree())

    # Add date
    if not date is None:
        ax.text(0.8, 1.02, "{}".format("Date : " + str(date)),
                transform=ax.transAxes, fontdict=dict(color="black", size=14))

    # Fix lats and lons to the given lons_lats instead of some reduced size based on the values of data
    if not adjustBorder:
        ax.set_extent([np.min(lons_lats[:, :, 0]), np.max(lons_lats[:, :, 0]),
                       np.min(lons_lats[:, :, 1]), np.max(lons_lats[:, :, 1])],
                      crs=ccrs.PlateCarree())

    if not maxVal is None and not minVal is None:
        cs.set_clim(minVal, maxVal)
    else:
        cs.set_clim(np.nanmin(data), np.nanmax(data))

    # Add Colorbar
    cbar = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)

    if not unit is None:
        cbar.ax.set_ylabel(unit, fontdict=dict(color="black", size=16))

    if not key is None:
        ax.text(0.0, 1.02, "{}".format("Quantity: " + key),
                transform=ax.transAxes, fontdict=dict(color="black", size=14))
    plt.show()


def clean_up_artists(axis, artist_list):
    """
    Try to remove the artists stored in the artist list belonging to the 'axis'.
     axis: clean artists belonging to these axis
     artist_list: list of artist to remove
    return: nothing
    """
    for artist in artist_list:
        try:
            # fist attempt: try to remove collection of contours for instance
            while artist.collections:
                for col in artist.collections:
                    artist.collections.remove(col)
                    try:
                        axis.collections.remove(col)
                    except ValueError:
                        pass

                artist.collections = []
                axis.collections = []
        except AttributeError:
            pass

        # second attempt, try to remove the text
        try:
            artist.remove()
        except (AttributeError, ValueError):
            pass


def trim_axs(axs, N):
    """Little helper to reshape the axis list to have correct length..."""
    try:
        axs = axs.flat
        for ax in axs[N:]:
            ax.remove()
        return axs[:N]
    except:
        return[axs]


def update_plot(frame_index, data_list, lons, lats, fig, axis, n_cols, n_rows,
                number_of_contour_levels, v_min, v_max, changed_artists, d, keys, units):
    """
    Update the the contour plots of the time step 'frame_index'

     frame_index: Integer required by animation running from 0 to n_frames -1.
    For initialisation of the plot call 'update_plot' with frame_index = -1
     data_list: List with the 3D data (time x 2D data) per subplot
     lons: Longitude degrees
     lats: Latitude degrees
     fig: Reference to the figure
     axis: Reference to the list of axis with the axes per subplot
     n_cols: Number of subplot in horizontal direction
     n_rows: Number of subplot in vertical direction
     number_of_contour_levels: Number of contour levels
     v_min: Minimum global data value. If None take min(data) in the 2d dataset
     v_max: Maximum global data value. If None take the largest value in the 2d data set
     changed_artists: List of lists of artists which are updated between the time steps
    return: Changed_artists list
    """

    # Number of current subplot
    nr_subplot = 0

    for j_col in range(n_cols):
        for i_row in range(n_rows):

            ax = axis[nr_subplot]

            # In the first setup call, add and empty list which can hold the artists
            # belonging to the current axis
            if frame_index < 0:
                # initialise the changed artist list
                changed_artists.append(list())
            else:
                # Remove all artists in the list stored in changed_artists
                clean_up_artists(ax, changed_artists[nr_subplot])

            # Draw the field data from the multidimensional data array
            if isinstance(data_list[nr_subplot], np.ma.core.MaskedArray):
                if len(data_list[nr_subplot].shape) == 3:
                    data_2d = data_list[nr_subplot][frame_index, :, :]
                else:
                    data_2d = data_list[nr_subplot][frame_index, 0]
            elif isinstance(data_list[nr_subplot], np.ndarray):
                data_2d = data_list[nr_subplot][frame_index]
            else:
                print("Unhandled data type "+str(type(data_list[nr_subplot])))

            # Set map with coastlines and borders (uncomment for quicker processing...)
            if frame_index < 0:
                ax.coastlines(resolution='10m')
                ax.add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land', '10m'),
                               linestyle=':', facecolor='none', edgecolor='black')
                ax.add_feature(cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '10m'),
                               facecolor='none', edgecolor='blue')

            # Gridlines require an update at every iteration and slow down the simulation a lot.
            # Uncomment to have gridlines
            # gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,\
            #                    linewidth=1, color='gray', alpha=0.5, linestyle='--')
            #gl.xformatter = LONGITUDE_FORMATTER
            #gl.yformatter = LATITUDE_FORMATTER
            #gl.xlabels_top = False
            #gl.ylabels_right = False

            # Remove redundant axes
            # if i_row == n_rows-1:
            #    gl.ylabels_left = False

            # if j_col == 0:
            #    gl.xlabels_bottom = False

            # Set value boundaries
            if v_min is None:
                data_min = np.nanmin(data_2d)
            else:
                data_min = v_min[nr_subplot]
            if v_max is None:
                data_max = np.nanmax(data_2d)
            else:
                data_max = v_max[nr_subplot]

            # Set the contour levels belonging to this subplot
            levels = np.linspace(data_min, data_max,
                                 number_of_contour_levels+1, endpoint=True)

            # Cap data at set limits (otherwise values appear white)
            if ~np.isnan(data_2d).any():
                data_2d = data_2d*(data_2d <= v_max[nr_subplot])*(data_2d >= v_min[nr_subplot])\
                    + v_min[nr_subplot]*(data_2d <= v_min[nr_subplot])\
                    + v_max[nr_subplot] * \
                    (data_2d >= v_max[nr_subplot])

            # Create the contour plot
            if isinstance(lons, list):
                cs = ax.contourf(lons[nr_subplot], lats[nr_subplot], data_2d, levels=levels, cmap=cm.rainbow, zorder=0,
                                 transform=ccrs.PlateCarree())
                ax.set_extent([np.min(lons[nr_subplot]), np.max(lons[nr_subplot]),
                               np.min(lats[nr_subplot]), np.max(lats[nr_subplot])], crs=ccrs.PlateCarree())
            else:
                cs = ax.contourf(lons, lats, data_2d, levels=levels, cmap=cm.rainbow, zorder=0,
                                 transform=ccrs.PlateCarree())
            cs.set_clim(v_min[nr_subplot], v_max[nr_subplot])

            # Store the contours artists to the list of artists belonging to the current axis
            changed_artists[nr_subplot].append(cs)

            # Set the changing time counter in the top left subplot
            if i_row == n_rows-1 and j_col == 0:
                # Set a label to show the current time
                if isinstance(d[0], list):
                    time_text = ax.text(0.6, 1.05, "{}".format("Date : " + str(d[nr_subplot][frame_index])),
                                        transform=ax.transAxes, fontdict=dict(color="black", size=14))
                else:
                    time_text = ax.text(0.6, 1.05, "{}".format("Date : " + str(d[frame_index])),
                                        transform=ax.transAxes, fontdict=dict(color="black", size=14))

                # Store the artist of this label in the changed artist list
                changed_artists[nr_subplot].append(time_text)

            # Set the colourbar at initiation
            if frame_index < 0 and None not in v_max and None not in v_min:
                cbar = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.set_ylabel(units[nr_subplot])
                ax.text(0.0, 1.02, "{}".format("Quantity: " + keys[nr_subplot]),
                        transform=ax.transAxes, fontdict=dict(color="blue", size=12))

            if frame_index < -1 and isinstance(lons, list):
                ax.set_extent([np.min(lons[nr_subplot]), np.max(lons[nr_subplot]),
                               np.min(lats[nr_subplot]), np.max(lats[nr_subplot])], crs=ccrs.PlateCarree())

            nr_subplot += 1

    return changed_artists


class TimeSeries():

    def __init__(self, data, lons=None, lats=None, keys=None, units=None, d=None):
        '''
        Load data for animation
             data: List of masked arrays, where each one contains a certain quantity.
             lons: Longitude coordinates as vector or matrix with same sized lats matrix.
             lats: Latitude coordinates as vector or matrix with same sized lons matrix.
             keys: Quantity label (Oxygen, Nitrate, Phosphate, ...).
             units: Unit of the displayed quantity as a list of strings.
             d: List of time date entries. Will be plotted after date on the animation.
            return: nothing
        '''
        # Load content
        if isinstance(data, np.ndarray) and len(data.shape) == 4:
            self.data = [data[:, :, :, i]
                         for i in range(len(data[0, 0, 0, :]))]
        elif isinstance(data, np.ndarray) and len(data.shape) == 3:
            self.data = [data]
        else:
            self.data = data

        # If no coordinates are given create normal integer mesh
        if lons is None and lats is None:
            [self.lons, self.lats] = np.meshgrid(
                np.arange(data[0].shape[0]), np.arange(data[0].shape[1]))
        # Read in mesh as [:,:,0-1] mesh
        elif lats is None:
            self.lons = lons[:, :, 0]
            self.lats = lons[:, :, 1]
        elif lons is None:
            self.lons = lats[:, :, 0]
            self.lats = lats[:, :, 1]
        elif isinstance(lons, list) and isinstance(lats, list):
            self.lons = list()
            self.lats = list()
            [tmp1, tmp2] = np.meshgrid(lons[0], lats[0])
            self.lons.append(tmp1)
            self.lats.append(tmp2)
            [tmp1, tmp2] = np.meshgrid(lons[1], lats[1])
            self.lons.append(tmp1)
            self.lats.append(tmp2)
        else:
            # Check if given lons and lats are 2D or 1D -> then call meshgrid
            try:
                lons.shape[1]
                lats.shape[1]
                self.lons = lons
                self.lats = lats
            except IndexError:
                [self.lons, self.lats] = np.meshgrid(lons, lats)

        # Check if keys are given
        if keys is None:
            self.keys = list()
            for i in range(len(data)):
                self.keys.append(str(i))
        else:
            self.keys = keys

        # Check if units are given
        if units is None:
            self.units = list()
            for i in range(len(data)):
                self.units.append(" ")
        else:
            self.units = units

        # Check if a date is given
        if d is None:
            self.d = list()
            if isinstance(data, list):
                for i in range(len(data[0])):
                    self.d.append(i)
            else:
                for i in range(len(data)):
                    self.d.append(i)
        else:
            self.d = d

    def createAnimation(self, number_of_contour_levels=10, n_rows=2, n_cols=2,
                        max_data_value=None, min_data_value=None, start_frame=None, end_frame=None, skip_frames=None):
        '''
        Create animation with the data given at init
            number_of_contour_levels: Number of colours/contour levels to be displayed
            n_rows: Amount of subplots in a row
            n_cols: Amount of subplots in a column
            max_data_value: Maximal value of each plot e.g. [21, 380, 290, 18]
            min_data_value: Minimal value of each plot e.g. [0, 180, 0, 0]
            start_frame: Start from frame number e.g. 0
            end_frame: Start from frame number e.g. len(labels[:,0,0])
            skip_frames: Amount of frames -1 to skip between every displayed image e.g. 1 (no skipping)

            return: nothing
        '''

        # Check on given input
        if n_rows * n_cols != len(self.data):
            print("Amount of intended subplots ({}·{}) does not match the amount of data sets given ({})".format(
                n_rows, n_cols, len(self.data)))

        # Set frame control
        if start_frame is None or (start_frame > len(self.d) and len(self.d) != 2):
            start_frame = 0

        if end_frame is None or (end_frame > len(self.d) and len(self.d) != 2):
            end_frame = len(self.d)

        if skip_frames is None or (skip_frames > len(self.d) and len(self.d) != 2):
            skip_frames = 1  # 1 - no skipping

        frames = range(start_frame, end_frame, skip_frames)

        # For automatic scaling (Not recommended...)
        if max_data_value is None:
            max_data_value = [np.max(dataSet) for dataSet in self.data]
        if min_data_value is None:
            min_data_value = [np.max((0, np.min(dataSet)))
                              for dataSet in self.data]

        # Figure setup
        fig, axis = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True, sharey=True,
                                 figsize=(14, 8), subplot_kw={'projection': ccrs.Mercator()})

        axis = trim_axs(axis, n_rows*n_cols)
        fig.subplots_adjust(wspace=0.15, left=0.05, right=0.95)

        changed_artists = list()

        # create first image by calling update_plot with frame_index = -1
        changed_artists = update_plot(-1, self.data, self.lons, self.lats, fig, axis, n_cols, n_rows,
                                      number_of_contour_levels, min_data_value, max_data_value, changed_artists, self.d, self.keys, self.units)

        print("\nProcessing animation...")

        # Call the animation function. The fargs argument equals the parameter list of update_plot,
        # except the 'frame_index' parameter.
        self.ani = animation.FuncAnimation(fig, update_plot, frames=frames,
                                           fargs=(self.data, self.lons, self.lats, fig, axis, n_cols, n_rows,
                                                  number_of_contour_levels, min_data_value,
                                                  max_data_value, changed_artists, self.d, self.keys, self.units),
                                           blit=False, repeat=False)

        plt.show()

    def saveAnimation(self, fps=8, name='toLazytoName'):
        '''
        Save animation after computing it with createAnimation
             fps: Amount of frames displayed each second in the video e.g. 10.
             name: Name of the video. Is stored depending on the call path of the object.
            return: nothing
        '''
        if not hasattr(self, 'ani'):
            print(
                "No animation available. Please call createAnimation on the object before saving it.")
        else:
            print("Saving animation...")
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
            self.ani.save(name+'.mp4', writer=writer)


class SateliteTimeSeries(TimeSeries):
    '''
        Make an animation to compare satelite and normal data.
    '''

    def __init__(self, satData):
        from read_satelite_data import SateliteData
        data = [satData.data, satData.RefSet.data]
        lons = [satData.lons, satData.RefSet.lons]
        lats = [satData.lats, satData.RefSet.lats]
        keys = [satData.keys, satData.RefSet.keys]
        units = [satData.unit, satData.RefSet.unit]
        d = [satData.times, satData.RefSet.times]

        super().__init__(data, lons=lons, lats=lats, keys=keys, units=units, d=d)


def timeClustersVisualization(labels=None, data_points_per_year=12, n_clusters=4):

    label_matrix = np.full((data_points_per_year, n_clusters), 0)

    for i in range(len(labels)):
        label_matrix[i % data_points_per_year, int(labels[i])] += 1

    f, subplts = plt.subplots(n_clusters, 1)

    year_range = range(0, data_points_per_year, 1)
    for i in range(n_clusters):
        subplts[i].plot(year_range, label_matrix[:, i])
    plt.show()
    return



# Example on how to use the visualization on raw data
# When calling the class from a separate file import the class as:
#
# from visualization import TimeSeries
#


def main():

    datasets = []

    # Open the four data sets
    chl_path = os.path.abspath('MetO-NWS-BIO-dm-CHL.nc')
    datasets.append(Dataset(chl_path, mode='r'))
    doxy_path = os.path.abspath('MetO-NWS-BIO-dm-DOXY.nc')
    datasets.append(Dataset(doxy_path, mode='r'))
    nitr_path = os.path.abspath('MetO-NWS-BIO-dm-NITR.nc')
    datasets.append(Dataset(nitr_path, mode='r'))
    phos_path = os.path.abspath('MetO-NWS-BIO-dm-PHOS.nc')
    datasets.append(Dataset(phos_path, mode='r'))

    # Read quantity types (O2, PO4...) and note their units
    keys = list()
    keys.append(list(datasets[0].variables)[1])
    keys.append(list(datasets[1].variables)[0])
    keys.append(list(datasets[2].variables)[0])
    keys.append(list(datasets[3].variables)[1])
    units = ["$mg/m^3$", "$mmol/m^3$", "$mmol/m^3$", "$mmol/m^3$"]

    # Read the measurment dates
    time = datasets[0].variables['time']
    jd = netCDF4.num2date(time[:], time.units)
    d = list()
    for dd in jd:
        d.append(dt.date(dd.year, dd.month, dd.day))

    # Read position data and transform it to a meshgrid
    lons = datasets[0].variables['longitude'][:]
    lats = datasets[0].variables['latitude'][:]

    # It is not necessary to calculate the meshgrid, the class can take both
    # matrix or vector versions of the latitudes and longitudes...
    # lons, lats = np.meshgrid(lons,lats)

    # Load data and close document
    data = []
    for i in range(4):
        data.append(datasets[i].variables[keys[i]][:])
        datasets[i].close()

    # Or load precomputed data
    # with np.load('model_data.npz') as m:
    #    data = m['matrix']
    # with np.load('lons_lats.npz') as ll:
    #    lons = ll['lons_lats']
    #lats = None

    #max_data_value = [1, 1, 1, 1]
    #min_data_value = [0, 0, 0, 0]

    # Read data
    myAnimation = TimeSeries(data, lons=lons, lats=lats,
                             keys=keys, units=units, d=d)

    # Set visualization parameters
    # User defined value ranges (colour)
    max_data_value = [10, 360, 220, 6]
    min_data_value = [0, 220, 0, 0]

    # Create animation
    myAnimation.createAnimation(number_of_contour_levels=10, n_rows=2, n_cols=2,
                                max_data_value=max_data_value, min_data_value=min_data_value, start_frame=1000,
                                end_frame=2000, skip_frames=10)

    # Save animation and view
    # Note that the playback speed of the animation shown via Python might
    # not be the same as the one of the stored video (depends on the GPU)
    myAnimation.saveAnimation(fps=8, name='toLazytoName')


# Execute main only if the script is run directly
if __name__ == "__main__":
    main()
