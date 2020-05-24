import numpy as np
from scipy import stats
from clustering import average_data, timestep_clustering, sort_clusters, clustering
from visualization import geographic_plot, timeseries_plot, timeClustersVisualization
import matplotlib.pyplot as plt
from itertools import compress
from silhouette import silhouette_plot
import pickle


def generate_yearly_data():
    '''
    Generates the yearly data
    '''
    with np.load('model_data.npz') as m:
        matrix = m['matrix']
    av_matrix = average_data(matrix=matrix, delta_t=366)
    np.savez_compressed('av_model_dataYearly.npz', matrix=av_matrix)

    return av_matrix


def region_calculation(n_regions=4, show_silhouette=True):
    '''
    Generates the regions

    n_regions:          number of regions
    show_silhouette:    default True
    '''
    try:
        with np.load('av_model_dataYearly.npz') as av_m:
            av_matrix = av_m['matrix']
    except:
        av_matrix = generate_yearly_data()

    with np.load('lons_lats.npz') as ll:
        lons_lats = ll['lons_lats']

    labels = np.full(av_matrix.shape[:-1], np.nan)
    silhouette_scores = np.full(av_matrix.shape[0], np.nan)
    for i in range(av_matrix.shape[0]):
        cl, labels[i, :, :], cl_sizes, silhouette_scores[i] = timestep_clustering(
            matrix=av_matrix, timestep=i, mode="kmeans", n_clusters=n_regions, silhouette=False, verbose=False)
        labels[i, :, :] = sort_clusters(
            labels=labels[i, :, :], cluster_sizes=cl_sizes)

    region_labels = np.full(labels.shape[1:], np.nan)

    for i in range(region_labels.shape[0]):
        for j in range(region_labels.shape[1]):
            region_labels[i, j] = stats.mode(labels[:, i, j])[0]

    geographic_plot(data=region_labels,
                    lons_lats=lons_lats, levels=n_regions-1)
    print(silhouette_scores)

    print('Silhouette Average: ', np.mean(silhouette_scores))
    if show_silhouette:
        plt.hist(silhouette_scores)
        plt.show()

    return region_labels


def average_by_region(matrix=None, chemical=0, r_labels=None, n_regions=4):
    '''
    Generates the data of a chemical taking average by region (see region_calculation)

    matrix:     data matrix
    chemical:   0: CHL
                1: DOXY
                2: NITR
                3: PHOS
    r_labels:   region labels, different number for different region
    n_regiond:  number of regions
    '''
    data = np.full((matrix.shape[0], n_regions), np.nan)

    for i in range(matrix.shape[0]):
        d = [[0, 0]] * n_regions
        for j in range(r_labels.shape[0]):
            for k in range(r_labels.shape[1]):
                if not np.isnan(r_labels[j][k]):
                    d[int(r_labels[j, k])][0] += matrix[i, j, k, chemical]
                    d[int(r_labels[j, k])][1] += 1
        for region in range(n_regions):
            data[i, region] = d[region][0] / d[region][1]

    return data


def main():
    n_regions = 4

    # Loading region data and calculate them if not saved
    try:
        with np.load('region_labels.npz') as r_labels:
            region_labels = r_labels['matrix']
    except:
        region_labels = region_calculation(
            n_regions=n_regions, show_silhouette=True)
        np.savez_compressed('region_labels.npz', matrix=region_labels)

    print('Fetching Data...')
    with np.load('av_model_data30.npz') as m:
        av_matrix = m['matrix']
    with open("datetimes.txt", "rb") as fp:
        dates = pickle.load(fp)
    print('Finished Fetching Data')

    # Clustering parameters
    mode = 'kmeans'
    n_clusters = [12]

    data = []
    s_avg = []

    # Keeping relevant dates
    new_d = []
    for i in range(int(len(dates)/30.4325)):
        new_d.append(dates[int(i * 30.4325)])

    # Clustering with regional average by chemical
    for i in range(4):
        for n in n_clusters:
            data.append(average_by_region(matrix=av_matrix, chemical=i,
                                          r_labels=region_labels, n_regions=n_regions))

            # Clustering
            if mode == 'kmeans':
                clustered_data = clustering(
                    data=data[i], n_clusters=n, mode='kmeans', verbose=False)
            elif mode == 'hierarchical':
                clustered_data = clustering(
                    data=data[i], n_clusters=n, mode='hierarchical', verbose=False)

            print("The " + str(n) + " cluster sizes are:")
            cluster_sizes = [len(list(compress(data[i], clustered_data.labels_ == cluster)))
                             for cluster in range(n)]
            print(cluster_sizes)

            s_avg.append(silhouette_plot(labels=clustered_data.labels_,
                                         data=data[i], plotGraph=False, n_clusters=n))

            # timeseries_plot(data=clustered_data.labels_, t=new_d)
            timeClustersVisualization(
                labels=clustered_data.labels_, data_points_per_year=12, n_clusters=n)
    print(s_avg)


if __name__ == "__main__":
    main()
