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
    try to remove the artists stored in the artist list belonging to the 'axis'.
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


def update_plot(frame_index, data_list, lons, lats, fig, axis, n_cols, n_rows,\
                    number_of_contour_levels, v_min, v_max,changed_artists):
    """
    Update the the contour plots of the time step 'frame_index'

    :param frame_index: integer required by animation running from 0 to n_frames -1.
    For initialisation of the plot call 'update_plot' with frame_index = -1
    :param data_list: list with the 3D data (time x 2D data) per subplot
    :param lons: Longitude degrees
    :param lats: Latitude degrees
    :param fig: reference to the figure
    :param axis: reference to the list of axis with the axes per subplot
    :param n_cols: number of subplot in horizontal direction
    :param n_rows: number of subplot in vertical direction
    :param number_of_contour_levels: number of contour levels
    :param v_min: minimum global data value. If None take min(data) in the 2d dataset
    :param v_max: maximum global data value. If None take the largest value in the 2d data set
    :param changed_artists: list of lists of artists which are updated between the time steps
    :return: the changed_artists list
    """
    # Constant data specifications
    global d, keys, units

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
            data_2d = data_list[nr_subplot][frame_index,0]

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
            levels = np.linspace(data_min, data_max, number_of_contour_levels+1, endpoint=True)

            # Create the contour plot
            cs = ax.contourf(lons, lats, data_2d, levels=levels, cmap=cm.rainbow, zorder=0)
            cs.cmap.set_under("k")
            cs.cmap.set_over("k")
            cs.set_clim(v_min[nr_subplot], v_max[nr_subplot])

            # Store the contours artists to the list of artists belonging to the current axis
            changed_artists[nr_subplot].append(cs)

            # Set some grid lines on top of the contours
            ax.xaxis.grid(True, zorder=0, color="black", linewidth=0.5, linestyle='--')
            ax.yaxis.grid(True, zorder=0, color="black", linewidth=0.5, linestyle='--')

            # Set the x and y label on the bottom row and left column respectively
            if i_row == n_rows - 1:
                ax.set_xlabel(r"Longitude ")
            if j_col == 0:
                ax.set_ylabel(r"Latitude ")

            # Set the changing time counter in the top left subplot
            if i_row == 0 and j_col == 1:
                # Set a label to show the current time
                time_text = ax.text(0.6, 1.15, "{}".format("Date : "+ str(d[frame_index])),\
                                transform=ax.transAxes, fontdict=dict(color="black", size=14))

                # Store the artist of this label in the changed artist list
                changed_artists[nr_subplot].append(time_text)

            # Set the colourbar at initiation
            if frame_index < 0 and None not in v_max and None not in v_min:
                cbar = fig.colorbar(cs, ax=ax)
                cbar.ax.set_ylabel(units[nr_subplot])
                ax.text(0.0, 1.02, "{}".format("Quantity: "+ keys[nr_subplot]),\
                           transform=ax.transAxes, fontdict=dict(color="blue", size=12))

            nr_subplot += 1

    return changed_artists




def main():
    global d, keys, units
    datasets = []

    # Open the four data sets
    chl_path = os.path.abspath('MetO-NWS-BIO-dm-CHL.nc')
    datasets.append(Dataset(chl_path, mode = 'r'))
    doxy_path = os.path.abspath('MetO-NWS-BIO-dm-DOXY.nc')
    datasets.append(Dataset(doxy_path, mode = 'r'))
    nitr_path = os.path.abspath('MetO-NWS-BIO-dm-NITR.nc')
    datasets.append(Dataset(nitr_path, mode = 'r'))
    phos_path = os.path.abspath('MetO-NWS-BIO-dm-PHOS.nc')
    datasets.append(Dataset(phos_path, mode = 'r'))

    # Read quantity types (O2, PO4...) and note their units
    keys = list()
    keys.append(list(datasets[0].variables)[1])
    keys.append(list(datasets[1].variables)[0])
    keys.append(list(datasets[2].variables)[0])
    keys.append(list(datasets[3].variables)[1])
    units = ["mg/m^3","mmol/m^3","mmol/m^3","mmol/m^3"]

    # Read the measurment dates
    time = datasets[0].variables['time']
    jd = netCDF4.num2date(time[:],time.units)
    d = list()
    for dd in jd:
        d.append(dt.date(dd.year, dd.month, dd.day))

    # Read position data and transform it to a meshgrid
    lons = datasets[0].variables['longitude'][:]
    lats = datasets[0].variables['latitude'][:]
    lons, lats = np.meshgrid(lons,lats)

    # Load data and close document
    data = []
    for i in range(4):
        data.append(datasets[i].variables[keys[i]][:])

    for i in range(4):
        datasets[i].close()


    # Define values for animation
    n_pixels_x,n_pixels_y = lons.shape
    number_of_contour_levels = 20
    delay_of_frames = 1 # Additional delay between frames
    n_rows = 2  # number of subplot rows
    n_cols = 2  # number of subplot columns
    start_frame = 0
    end_frame = len(d)
    skip_frames = 24 # 1 - no skipping

    # For automatic scaling uncomment this ()
    #max_data_value = [np.max(dataSet) for dataSet in data]
    #min_data_value = [np.max((0,np.min(dataSet))) for dataSet in data]

    # User defined value ranges (colour)
    max_data_value = [21, 380, 290, 18]
    min_data_value = [0, 180, 0, 0]

    # Figure setup
    fig, axis = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True, sharey=True,\
       figsize=(12,8))
    #, subplot_kw={'projection': ccrs.Mercator()}
    ## TODO: No compatibility of the projections (maps, borders) with animation
    
    fig.subplots_adjust(wspace=0.05, left=0.08, right=0.98)

    ## TODO: No compatibility of the projections (maps, borders) with animation
    #axis[0][0].stock_img()
    #axis[0][0].coastlines(resolution='10m')
    #axis[0][0].add_feature(cfeature.BORDERS)

    #axis[0][1].stock_img()
    #axis[0][1].coastlines(resolution='10m')
    #axis[0][1].add_feature(cfeature.BORDERS)

    #axis[1][0].stock_img()
    #axis[1][0].coastlines(resolution='10m')
    #axis[1][0].add_feature(cfeature.BORDERS)

    #axis[1][1].stock_img()
    #axis[1][1].coastlines(resolution='10m')
    #axis[1][1].add_feature(cfeature.BORDERS)

    changed_artists = list()

    # create first image by calling update_plot with frame_index = -1
    changed_artists = update_plot(-1, data, lons, lats, fig, axis, n_cols, n_rows,\
            number_of_contour_levels, min_data_value, max_data_value, changed_artists)

    # Call the animation function. The fargs argument equals the parameter list of update_plot,
    # except the 'frame_index' parameter.
    ani = animation.FuncAnimation(fig, update_plot,frames=range(start_frame,end_frame,skip_frames),
                                  fargs=(data, lons, lats, fig, axis, n_cols, n_rows,\
                                     number_of_contour_levels, min_data_value,\
                                         max_data_value, changed_artists),\
                                  interval=delay_of_frames, blit=False, repeat=False)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
    ani.save('Fulldata10fps.mp4', writer=writer)
    plt.show()

if __name__ == "__main__":
    main()