import sklearn.cluster as cluster
import numpy as np
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import compress
import sys
import pickle
from visualization import TimeSeries, geographic_plot
from silhouette import silhouette_plot, elbowPlot, plot_dendrogram


def single_chemical_clustering(matrix=None, chemical=None, mode='kmeans', n_clusters=10, dbscan_eps=3, metric='euclidean', silhouette=False, verbose=True):
    '''
    This function clusters spatially the data of a certain chemical through time and returns the clustered data
    and the labels organized spatially.

    matrix:     the data through time or the data at a particular timestep
    chemical:   if None the matrix is already given for the selected chemical
                if not None is the index of chemical to cluster
                0: CHL
                1: DOXY
                2: NITR
                3: PHOS
    mode:       clustering mode (kmeans, dbscan, hierarchical)
    n_clusters: for kmeans and hierarchical, is the number of clusters
    dbscan_eps: for dbscan, the maximal neighboring distance
    metric:     for dbscan, the metric used for distance calculations
    silhouette: plots the silohuette of the clusters (default False)
    verbose:    displays additional information while cluatering
    '''
    data = None

    if chemical == None:
        data = np.asarray(matrix)
    else:
        if chemical >= matrix.shape[3]:
            print("Chemical Index not valid!")
            return
        else:
            data = matrix[:, :, :, chemical]
    # Straighten-out data for clustering
    straight_data = []
    coordinates = []

    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            if (~np.isnan(data[:, i, j])).all():
                d = data[:, i, j].tolist()
                coordinates.append([i, j])
                straight_data.append(d)

    straight_data = straight_data[1:]

    # Clustering
    if mode == 'kmeans':
        clustered_data = clustering(
            data=straight_data, n_clusters=n_clusters, mode='kmeans', verbose=verbose)
    elif mode == 'dbscan':
        clustered_data = clustering(
            data=straight_data, mode='dbscan', metric=metric, dbscan_epsilon=dbscan_eps, verbose=verbose)
        n_clusters = max(clustered_data.labels_) + 1
    elif mode == 'hierarchical':
        clustered_data = clustering(
            data=straight_data, n_clusters=n_clusters, mode='hierarchical', verbose=verbose)

    cluster_sizes = [len(list(compress(straight_data, clustered_data.labels_ == i)))
                     for i in range(n_clusters)]
    if verbose:
        print("The " + str(n_clusters) + " cluster sizes are:")
        print(cluster_sizes)

    # Saving lables in a spatial martix
    labels = np.full(data.shape[1:], np.nan)
    straight_labels = np.full(len(straight_data), np.nan)

    for i in range(len(straight_data)):
        if clustered_data.labels_[i] >= 0:
            labels[coordinates[i][0], coordinates[i][1]
                   ] = clustered_data.labels_[i]
            straight_labels[i] = clustered_data.labels_[i]

    s_avg = 0
    s_avg = silhouette_plot(labels=straight_labels, data=straight_data,
                            plotGraph=silhouette, n_clusters=n_clusters)

    del straight_data

    return clustered_data, labels, cluster_sizes, s_avg


def timestep_clustering(matrix=None, timestep=None, mode='kmeans', n_clusters=10, dbscan_eps=3, metric='euclidean', silhouette=False, verbose=True, **kwargs):
    '''
    This function clusters spatially the data at a certain timestep and returns the clustered data
    and the labels organized spatially.

    matrix:     the data through time or the data at a particular timestep
    timestep:   if None: the matrix is already given at a single timestep
                if not None: it is the timestep to cluster
    mode:       clustering mode (kmeans, dbscan, hierarchical)
    n_clusters: for kmeans and hierarchical, is the number of clusters
    dbscan_eps: for dbscan, the maximal neighboring distance
    metric:     for dbscan, the metric used for distance calculations
    silhouette: plots the silohuette of the clusters (default False)
    verbose:    displays additional information while cluatering
    '''
    m_shape = len(matrix.shape)
    data = None
    if timestep == None:
        data = np.asarray(matrix)
    else:
        if timestep >= matrix.shape[0]:
            print("Timestep not valid!")
            return
        else:
            if m_shape == 4:
                data = matrix[timestep, :, :, :]
            else:
                data = matrix[timestep, :, :]

    # Straighten-out data for clustering
    straight_data = []
    coordinates = []

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if m_shape == 4:
                not_nan = ~np.isnan(data[i, j, :]).all()
            else:
                not_nan = not data.mask[i, j]
            if not_nan:
                if m_shape == 4:
                    d = data[i, j, :].tolist()
                else:
                    d = [data[i, j], 0]
                coordinates.append([i, j])
                straight_data.append(d)

    # Clustering
    if mode == 'kmeans':
        clustered_data = clustering(
            data=straight_data, n_clusters=n_clusters, mode='kmeans', verbose=verbose, **kwargs)
    elif mode == 'dbscan':
        clustered_data = clustering(
            data=straight_data, mode='dbscan', metric=metric, dbscan_epsilon=dbscan_eps, verbose=verbose, **kwargs)
        n_clusters = max(clustered_data.labels_) + 1
    elif mode == 'hierarchical':
        clustered_data = clustering(
            data=straight_data, n_clusters=n_clusters, mode='hierarchical', verbose=verbose, **kwargs)

    cluster_sizes = [len(list(compress(straight_data, clustered_data.labels_ == i)))
                     for i in range(n_clusters)]
    if verbose:
        print("The " + str(n_clusters) + " cluster sizes are:")
        print(cluster_sizes)

    # Saving lables in a spatial martix
    labels = np.full(data.shape[0:2], np.nan)
    straight_labels = np.full(len(straight_data), np.nan)

    for i in range(len(straight_data)):
        if clustered_data.labels_[i] >= 0:
            labels[coordinates[i][0], coordinates[i][1]
                   ] = clustered_data.labels_[i]
            straight_labels[i] = clustered_data.labels_[i]

    s_avg = 0
    s_avg = silhouette_plot(labels=straight_labels, data=straight_data,
                            plotGraph=silhouette, n_clusters=n_clusters)

    del straight_data

    return clustered_data, labels, cluster_sizes, s_avg


def timewise_clustering(matrix=None, location=None, chemicals=[True, True, True, True], mode='kmeans', n_clusters=10, dbscan_eps=3, metric='euclidean', silhouette=False, verbose=False, **kwargs):
    '''
    This function clusters the data of the selected chemicals timewise 
    and returns the clustered data and the labels.

    matrix:     the data through time
    location:   if None: it clusters the dayly average
                if not None: it clusters at a specific location
    chemicals:  is the index of chemical to cluster
                0: CHL
                1: DOXY
                2: NITR
                3: PHOS
    mode:       clustering mode (kmeans, dbscan, hierarchical)
    n_clusters: for kmeans and hierarchical, is the number of clusters
    dbscan_eps: for dbscan, the maximal neighboring distance
    metric:     for dbscan, the metric used for distance calculations
    silhouette: plots the silohuette of the clusters (default False)
    verbose:    displays additional information while clustering
    '''
    data = None
    if location != None:
        data = np.asarray(matrix)
        straight_data = []

        for time in range(data.shape[0]):
            d = data[time, location[0], location[1], chemicals].tolist()
            straight_data.append(d)
    else:
        n_chemicals = sum(np.multiply(chemicals, 1))
        straight_data = []
        # np.full((matrix.shape[0], n_chemicals), np.nan)
        for t in range(matrix.shape[0]):
            d = []
            for chem in range(matrix.shape[3]):
                if chemicals[chem]:
                    temp_data = matrix[t, :, :, chem]
                    not_nan_indexes = ~np.isnan(temp_data)
                    n_not_nans = np.sum(np.multiply(not_nan_indexes, 1))
                    d.append(np.sum(temp_data[not_nan_indexes])/n_not_nans)
            straight_data.append(d)
    # Clustering
    if mode == 'kmeans':
        clustered_data = clustering(
            data=straight_data, n_clusters=n_clusters, mode='kmeans', **kwargs)
    elif mode == 'dbscan':
        clustered_data = clustering(
            data=straight_data, mode='dbscan', metric=metric, dbscan_epsilon=dbscan_eps, **kwargs)
        n_clusters = max(clustered_data.labels_) + 1
    elif mode == 'hierarchical':
        clustered_data = clustering(
            data=straight_data, n_clusters=n_clusters, mode='hierarchical', **kwargs)

    if not n_clusters is None:

        cluster_sizes = [len(list(compress(straight_data, clustered_data.labels_ == i)))
                         for i in range(n_clusters)]
        if verbose:
            print("The " + str(n_clusters) + " cluster sizes are:")
            print(cluster_sizes)

        # Saving lables in a spatial martix
        labels = np.full(len(straight_data), np.nan)

        for i in range(len(straight_data)):
            if clustered_data.labels_[i] >= 0:
                labels[i] = clustered_data.labels_[i]

        s_avg = silhouette_plot(labels=labels, data=straight_data,
                                plotGraph=silhouette, n_clusters=n_clusters)
    else:
        labels = []
        cluster_sizes = []
        s_avg = []

    del straight_data

    return clustered_data, labels, cluster_sizes, s_avg


def clustering(data=None, n_clusters=10, mode='kmeans', metric='euclidean', dbscan_epsilon=1, verbose=True, **kwargs):
    '''
    This function clusters the received data according to the parameters given

    data:       rectangular matrix to be clustered, shape=(n_samples, n_features)
    mode:       clustering mode (kmeans, dbscan, hierarchical)
    n_clusters: for kmeans and hierarchical, the number of clusters
    dbscan_eps: for dbscan, the maximal neighboring distance
    metric:     for dbscan, the metric used for distance calculations
    '''
    if verbose:
        print("Starting the Clustering Procedure, using mode: " + mode)

    if mode == 'kmeans':
        clusterer = cluster.KMeans(
            n_clusters=n_clusters, init='k-means++', **kwargs)
        clusterer.fit(data)
    elif mode == 'dbscan':
        clusterer = cluster.DBSCAN(eps=dbscan_epsilon, metric=metric, **kwargs)
        clusterer.fit(data)
    elif mode == 'hierarchical':
        clusterer = cluster.AgglomerativeClustering(
            n_clusters=n_clusters, **kwargs)
        clusterer.fit(data)

    if verbose:
        print("Finished Clustering.")
    return clusterer


def average_data(matrix=None, delta_t=10):
    '''
    This function averages the data through time

    matrix:     data through time, space and chemicals
    chemicals:  boolean array that represent which chemicals are being averaged;
                as default, all of them are averaged
    delta_t:    time period over which to average the data
    '''
    m_shape = len(matrix.shape)
    n_chemicals = matrix.shape[-1] if m_shape==4 else 0
    time_steps = int(matrix.shape[0]/delta_t)
    if m_shape == 4:
        data = np.full(
            (time_steps, matrix.shape[1], matrix.shape[2], matrix.shape[3]), np.nan)

        print("Starting Averaging Procedure:")
        sys.stdout.write("[%s]" % (" " * 10))
        sys.stdout.flush()
        sys.stdout.write("\b" * (10+1))
        count = 0.1
        for t in range(time_steps):
            layer = np.zeros((matrix.shape[1], matrix.shape[2], n_chemicals))
            for i in range(matrix.shape[1]):
                for j in range(matrix.shape[2]):
                    t1 = int(t*delta_t)
                    t2 = int(min((t + 1)*delta_t, matrix.shape[0] - 1))
                    for chem in range(n_chemicals):
                        temp_data = matrix[t1:t2, i, j, chem]
                        not_nan_indexes = ~np.isnan(temp_data)
                        n_not_nans = sum(np.multiply(not_nan_indexes, 1))
                        if n_not_nans > 0:
                            d = np.sum(temp_data[not_nan_indexes])/n_not_nans
                            layer[i, j, chem] = d
                        else:
                            layer[i, j, chem] = np.nan
            data[t, :, :, :] = layer
            if t / time_steps >= count:
                sys.stdout.write(chr(9608))
                sys.stdout.flush()
                count += 0.1

        sys.stdout.write(chr(9608))
        sys.stdout.flush()
        print("\nFinished Averaging.")

        return data[:, :, :, :]
    else:
        data = np.full(
            (time_steps, matrix.shape[1], matrix.shape[2]), np.nan)

        print("Starting Averaging Procedure:")
        sys.stdout.write("[%s]" % (" " * 10))
        sys.stdout.flush()
        sys.stdout.write("\b" * (10+1))
        count = 0.1
        for t in range(time_steps):
            layer = np.zeros((matrix.shape[1], matrix.shape[2]))
            for i in range(matrix.shape[1]):
                for j in range(matrix.shape[2]):
                    t1 = int(t*delta_t)
                    t2 = int(min((t + 1)*delta_t, matrix.shape[0] - 1))
                    temp_data = matrix[t1:t2, i, j]
                    mask = ~temp_data.mask
                    n_not_nans = sum(np.multiply(mask, 1))
                    if n_not_nans > 0:
                        d = np.sum(temp_data[mask])/n_not_nans
                        layer[i, j] = d
                    else:
                        layer[i, j] = np.nan
            data[t, :, :] = layer
            if t / time_steps >= count:
                sys.stdout.write(chr(9608))
                sys.stdout.flush()
                count += 0.1

        sys.stdout.write(chr(9608))
        sys.stdout.flush()
        print("\nFinished Averaging.")

        return data[:, :, :]


def sort_clusters(labels=None, cluster_sizes=[]):
    n_clusters = len(cluster_sizes)
    new_labels = []
    for i in range(n_clusters):
        position = 0
        for j in range(n_clusters):
            if cluster_sizes[i] > cluster_sizes[j]:
                position += 1
        new_labels.append(position)

    for i in range(n_clusters):
        labels = np.where(labels == i, new_labels[i] + n_clusters, labels)

    return labels - n_clusters


def clustervalues(matrix, labels, lons_lats, d, lon = 8.6865, lat = 54.025, chem = ['no3','po4']):
    '''
    Function to plot the yearly mean of two cuantities of a certain cluster. The preset values correspong
    to the river estuary of the Elba, Weser and Rhine rivers.

    matrix:     Matrix [timesteps, lons, lats, chem] containing the unmodified model values.
    labels:     Labels of the clustering algorithm.
    lons_lats:  Spatial coordinates of the form [:,:,0:1].
    lon:        Longitude of a point within the cluster of interest.
    lat:        Latitude of a point within the cluster of interest.
    chems:      The chemicals that shall be plotted e.g. ['no3','po4']
    '''
    chems = list()

    for c in chem:
        if c == 'CHL' or c == 'chl' or c == 'Chl':
            chems.append((0,r'$Chl~[\frac{mg}{m^3}]$'))
        elif c == 'O2' or c == 'o2' or c == 'O_2':
            chems.append((1,r'$O_2~[\frac{mmol}{m^3}]$'))
        elif c == 'NO3' or c == 'no3' or c == 'no_3':
            chems.append((2,r'$NO_3~[\frac{mmol}{m^3}]$'))
        elif c == 'PO4' or c == 'po4' or c == 'po_4':
            chems.append((3,r'$PO_4~[\frac{mmol}{m^3}]$'))




    from read_satelite_data import findClose
    latEstuary = findClose(lons_lats[:,0,1], 54.025, end = 'min')
    lonEstuary = findClose(lons_lats[0,:,0], 8.6865, end = 'min')

    label_estuary = labels[latEstuary,lonEstuary]

    n = len(d)
    no3 = list()
    po4 = list()
    dates = [1998]
    datesIdx = [1998]
    years = np.array([date.year for date in d])
    idx = np.where(labels == label_estuary)

    rawNO3 = matrix[:,:,:,chems[0][0]]
    rawPO4 = matrix[:,:,:,chems[1][0]]
    meanNO3 = np.zeros(n)
    meanPO4 = np.zeros(n)

    for k in range(n):

        datesIdx.append(d[k].year)

        if int(dates[-1]) < d[k].year:
            dates.append(d[k].year)

        tmp = rawNO3[k]
        meanNO3[k] = np.nanmean(tmp[idx])
        tmp = rawPO4[k]
        meanPO4[k] = np.nanmean(tmp[idx])
    
    datesIdx.pop(-1)
    datesIdx = np.array(datesIdx)

    for k in dates:
        no3.append(np.mean(meanNO3[np.where(datesIdx == k)]))
        po4.append(np.mean(meanPO4[np.where(datesIdx == k)]))

    no3 = np.array(no3)
    po4 = np.array(po4)

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Year', fontdict=dict(size=12))
    ax1.set_ylabel(chems[0][1], color=color, fontdict=dict(size=12))
    ax1.plot(dates, no3, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(dates[0::2])
    ax1.set_xticklabels(dates[0::2])
    ax1.set_xlim([np.min(dates[0::2]), np.max(dates[0::2])])

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(chems[1][1], color=color, fontdict=dict(size=12))  # we already handled the x-label with ax1
    ax2.plot(dates, po4, color = color)
    ax2.tick_params(axis = 'y', labelcolor = color)
    ax2.set_title("Mean annual concentrations in the estuary cluster", fontdict=dict(color="black", size=12))

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def main():
    # Loading already saved data (see save_data.py)
    print("Fetching data...")
    # with np.load('model_data.npz') as m:
    #     matrix = m['matrix']
    with open("datetimes.txt", "rb") as fp:   # Unpickling
        d = pickle.load(fp)
    with np.load('lons_lats.npz') as ll:
        lons_lats = ll['lons_lats']
    # Average the data through time (if needed)
    #av_matrix = average_data(matrix=matrix, delta_t=30.4325)
    #np.savez_compressed('av_model_data30.npz', matrix=av_matrix)
    with np.load('av_model_data30.npz') as av_m:
        av_matrix = av_m['matrix']
    print("Finished fetching data")

    # Clustering variables
    tstep = 50
    chem = 1
    n_clusters = 4
    dbscan_eps = 0.8

    # r_labels = region_clusters(data=av_matrix, lons_lats=lons_lats, n_clusters=4)

    # Uncomment one of the following to cluster

    # Clustering with kmeans

    cl, labels, cs, s_avg = timestep_clustering(matrix=av_matrix, timestep=tstep, mode="dbscan", dbscan_eps=dbscan_eps, silhouette=False)
    geographic_plot(data=labels, lons_lats = lons_lats, levels = len(cs) - 1)

    # cl, labels, cs, s_avg = single_chemical_clustering(
    #     matrix=av_matrix, chemical=chem, mode="kmeans", n_clusters=n_clusters, silhouette=True)

    # clusterNumbers = np.arange(2, 15, 1)
    # inertias = list()
    # silhouette = list()

    # for k in clusterNumbers:
    #     cl, labels, cs, s_avg = timestep_clustering(
    #         matrix=av_matrix, timestep=0, mode="kmeans", n_clusters=k, silhouette=False)
    #     inertias.append(cl.inertia_)
    #     silhouette.append(s_avg)

    # plt.plot(clusterNumbers, silhouette)
    # plt.show()

    # elbowPlot(inertiaVals=inertias, n_cluster=clusterNumbers)

    # Clustering with hierarchical/agglomeratative
    #cl, labels, cs, s_avg = timewise_clustering(matrix=av_matrix, mode="hierarchical", n_clusters=None, silhouette=False, distance_threshold=0)

    #plot_dendrogram(cl, truncate_mode='level', p=5)
    # from double_clustering import region_calculation
    # try:
    #     with np.load('region_labels.npz') as r_labels:
    #         region_labels = r_labels['matrix']
    # except:
    #     region_labels = region_calculation(
    #         n_regions=4, show_silhouette=True)
    #     np.savez_compressed('region_labels.npz', matrix=region_labels)

    #clustervalues(matrix, region_labels, lons_lats, d, lon = 8.6865, lat = 54.025, chem = ['no3','po4'])

    #cl, labels, cs, s_avg = timestep_clustering(matrix=av_matrix, timestep=tstep, mode="kmeans", n_clusters=n_clusters, silhouette=True)

    # plot_dendrogram(cl, truncate_mode='level', p=5)
    # geographic_plot(data=region_labels, lons_lats = lons_lats, levels = 4, minVal = 0, maxVal = 4, adjustBorder = False)

    # cl, labels, cs, s_avg = single_chemical_clustering(matrix=matrix, chemical=chem, mode="hierarchical", n_clusters=n_clusters)

    # Clustering with dbscan (kinda shit)
    # cl, labels, cs, s_avg = timestep_clustering(matrix=matrix, timestep=tstep, mode="dbscan", dbscan_eps=dbscan_eps)
    # cl, labels, cs, s_avg = single_chemical_clustering(
    #     matrix=matrix, chemical=chem, mode="dbscan", dbscan_eps=dbscan_eps)



    # Display and Save Animation
    # ts = TimeSeries(labels, lons_lats[:, :, 0], lons_lats[:, :, 1])
    # ts.createAnimation(max_data_value=[5, 5, 5, 5], min_data_value=[-1, -1, -1, -1], n_rows=1, n_cols=1)
    # ts.saveAnimation(name='clusters_Yearly_av')



    # Plot cluster labels geographically

    # geographic_plot(data=data, lons_lats=lons_lats, key = '', unit = '', date = '', minVal = np.nanmin(data), maxVal = np.nanmax(data), adjustBorder = False)

    # Plot cluster labels through time
    # cl, labels, s = timewise_clustering(matrix=av_matrix, n_clusters=n_clusters)

    # # Keeping relevant dates
    # new_d = []
    # for i in range(len(d)):
    #     if i % 366 == 0:
    #         new_d.append(d[i])

    # new_d.pop()

    # timeseries_plot(data=labels, t=new_d)


if __name__ == "__main__":
    main()
