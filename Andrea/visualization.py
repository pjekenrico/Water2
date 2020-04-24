# -*- coding: utf-8 -*-
import os
import netCDF4
from netCDF4 import Dataset
import numpy as np
import datetime as dt
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import cm
import string


def clean_up_artists(axis, artist_list):
    """
    Try to remove the artists stored in the artist list belonging to the 'axis'.
    :param axis: clean artists belonging to these axis
    :param artist_list: list of artist to remove
    :return: nothing
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


def update_plot(frame_index, data_list, lons, lats, fig, axis, n_cols, n_rows,
                number_of_contour_levels, v_min, v_max, changed_artists, d, keys, units):
    """
    Update the the contour plots of the time step 'frame_index'

    :param frame_index: Integer required by animation running from 0 to n_frames -1.
    For initialisation of the plot call 'update_plot' with frame_index = -1
    :param data_list: List with the 3D data (time x 2D data) per subplot
    :param lons: Longitude degrees
    :param lats: Latitude degrees
    :param fig: Reference to the figure
    :param axis: Reference to the list of axis with the axes per subplot
    :param n_cols: Number of subplot in horizontal direction
    :param n_rows: Number of subplot in vertical direction
    :param number_of_contour_levels: Number of contour levels
    :param v_min: Minimum global data value. If None take min(data) in the 2d dataset
    :param v_max: Maximum global data value. If None take the largest value in the 2d data set
    :param changed_artists: List of lists of artists which are updated between the time steps
    :return: Changed_artists list
    """

    # Number of current subplot
    nr_subplot = 0

    for j_col in range(n_cols):
        for i_row in range(n_rows):

            ax = axis[i_row][j_col]

            # In the first setup call, add and empty list which can hold the artists
            # belonging to the current axis
            if frame_index < 0:
                # initialise the changed artist list
                changed_artists.append(list())
            else:
                # Remove all artists in the list stored in changed_artists
                clean_up_artists(ax, changed_artists[nr_subplot])

            # Draw the field data from the multidimensional data array
            data_2d = data_list[frame_index, :, :, nr_subplot]

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

            # Create the contour plot
            cs = ax.contourf(lons, lats, data_2d, levels=levels,
                             cmap=cm.rainbow, zorder=0)
            cs.cmap.set_under("k")
            cs.cmap.set_over("k")
            cs.set_clim(v_min[nr_subplot], v_max[nr_subplot])

            # Store the contours artists to the list of artists belonging to the current axis
            changed_artists[nr_subplot].append(cs)

            # Set some grid lines on top of the contours
            ax.xaxis.grid(True, zorder=0, color="black",
                          linewidth=0.5, linestyle='--')
            ax.yaxis.grid(True, zorder=0, color="black",
                          linewidth=0.5, linestyle='--')

            # Set the x and y label on the bottom row and left column respectively
            if i_row == n_rows - 1:
                ax.set_xlabel(r"Longitude ")
            if j_col == 0:
                ax.set_ylabel(r"Latitude ")

            # Set the changing time counter in the top left subplot
            if i_row == 0 and j_col == 1:
                # Set a label to show the current time
                time_text = ax.text(0.6, 1.15, "{}".format("Date : " + str(d[frame_index])),
                                    transform=ax.transAxes, fontdict=dict(color="black", size=14))

                # Store the artist of this label in the changed artist list
                changed_artists[nr_subplot].append(time_text)

            # Set the colourbar at initiation
            if frame_index < 0 and None not in v_max and None not in v_min:
                cbar = fig.colorbar(cs, ax=ax)
                cbar.ax.set_ylabel(units[nr_subplot])
                ax.text(0.0, 1.02, "{}".format("Quantity: " + keys[nr_subplot]),
                        transform=ax.transAxes, fontdict=dict(color="blue", size=12))

            nr_subplot += 1

    return changed_artists


class TimeSeries():

    def __init__(self, data, lons, lats, keys=None, units=None, d=None):
        '''
        Load data for animation
            :param data: List of masked arrays, where each one contains a certain quantity.
            :param lons: Longitude coordinates as vector or matrix with same sized lats matrix.
            :param lats: Latitude coordinates as vector or matrix with same sized lons matrix.
            :param keys: Quantity label (Oxygen, Nitrate, Phosphate, ...).
            :param units: Unit of the displayed quantity as a list of strings.
            :param d: List of time date entries. Will be plotted after date on the animation.
            :return: nothing
        '''
        # Load content
        self.data = data
        self.keys = keys

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
            for i in range(len(data[0])):
                self.d.append(i)
        else:
            self.d = d

        print("Domain coordinates: " + str((np.min(self.lats), np.max(self.lats))
                                           ) + ", " + str((np.min(self.lons), np.max(self.lons))))
        print("Domain dimensions (lat, lon): " + str(self.lons.shape))
        print("Time frame: " + str(self.d[0]) + " - " + str(self.d[-1]))
        print("Number of time steps: " + str(len(self.d)))
    def createAnimation(self, number_of_contour_levels=10, n_rows=2, n_cols=2,
                        max_data_value=None, min_data_value=None, start_frame=None, end_frame=None, skip_frames=None):
        '''
        Create animation with the data given at init
            :param max_data_value: Maximal value of each plot e.g. [21, 380, 290, 18]
            :param min_data_value: Minimal value of each plot e.g. [0, 180, 0, 0]

            :param number_of_contour_levels: Number of colours/contour levels to be displayed
            :param n_rows: Amount of subplots in a row
            :param n_cols: Amount of subplots in a column
            :return: nothing
        '''

        # Image sizes
        n_pixels_x, n_pixels_y = self.lons.shape

        # Check on given input
        if n_rows * n_cols != len(self.data):
            print("Amount of intended subplots ({}·{}) does not match the amount of data sets given ({})".format(
                n_rows, n_cols, len(self.data)))

        # Set frame control
        if start_frame is None:
            start_frame = 0

        if end_frame is None:
            end_frame = len(self.d)

        if skip_frames is None:
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
                                 figsize=(12, 8))
        # , subplot_kw={'projection': ccrs.Mercator()}
        # TODO: No compatibility of the projections (maps, borders) with animation

        fig.subplots_adjust(wspace=0.05, left=0.08, right=0.98)

        # TODO: No compatibility of the projections (maps, borders) with animation
        # axis[0][0].stock_img()
        # axis[0][0].coastlines(resolution='10m')
        # axis[0][0].add_feature(cfeature.BORDERS)

        # axis[0][1].stock_img()
        # axis[0][1].coastlines(resolution='10m')
        # axis[0][1].add_feature(cfeature.BORDERS)

        # axis[1][0].stock_img()
        # axis[1][0].coastlines(resolution='10m')
        # axis[1][0].add_feature(cfeature.BORDERS)

        # axis[1][1].stock_img()
        # axis[1][1].coastlines(resolution='10m')
        # axis[1][1].add_feature(cfeature.BORDERS)

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
    def saveAnimation(self, fps=8, name='toLazytoName', showAnim=True):
        '''
        Save animation after computing it with createAnimation
            :param fps: Amount of frames displayed each second in the video e.g. 10.
            :param name: Name of the video. Is stored depending on the call path of the object.
            :return: nothing
        '''
        if not hasattr(self, 'ani'):
            print(
                "No animation available. Please call createAnimation on the object before saving it.")
        else:
            print("Saving animation...")
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
            self.ani.save(name+'.mp4', writer=writer)

            if showAnim:
                plt.show()


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
    units = ["mg/m^3", "mmol/m^3", "mmol/m^3", "mmol/m^3"]

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

    for i in range(4):
        datasets[i].close()

    # Read data
    myAnimation = TimeSeries(data, lons, lats, keys=keys, units=units, d=d)

    # Set visualization parameters

    # User defined value ranges (colour)
    max_data_value = [21, 380, 290, 18]
    min_data_value = [0, 180, 0, 0]

    # Create animation
    myAnimation.createAnimation(number_of_contour_levels=10, n_rows=2, n_cols=2,
                                max_data_value=max_data_value, min_data_value=min_data_value, start_frame=1000,
                                end_frame=2000, skip_frames=10)

    # Save animation and view
    # Note that the playback speed of the animation shown via Python might
    # not be the same as the one of the stored video (depends on the GPU)
    myAnimation.saveAnimation(fps=8, name='toLazytoName', showAnim=True)


# Execute main only if the script is run directly
if __name__ == "__main__":
    main()
