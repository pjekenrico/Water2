import numpy as np
from scipy import stats
from clustering import average_data, timestep_clustering, sort_clusters, clustering
from visualization import geographic_plot, timeseries_plot
import matplotlib.pyplot as plt
from itertools import compress
from silhouette import silhouette_plot
import pickle


def generate_yearly_data():
    with np.load('model_data.npz') as m:
        matrix = m['matrix']
    av_matrix = average_data(matrix=matrix, delta_t=366)
    np.savez_compressed('av_model_dataYearly.npz', matrix=av_matrix)

    return av_matrix


def region_clusters(data=None, lons_lats=None, n_regions=4, silhouette=False):
    labels = np.full(data.shape[:-1], np.nan)
    s_scores = np.full(data.shape[0], np.nan)
    for i in range(data.shape[0]):
        cl, labels[i, :, :], cl_sizes, s_scores[i] = timestep_clustering(
            matrix=data, timestep=i, mode="kmeans", n_clusters=n_regions, silhouette=silhouette, verbose=False)
        labels[i, :, :] = sort_clusters(
            labels=labels[i, :, :], cluster_sizes=cl_sizes)
        print(i)

    regions_labels = np.full(labels.shape[1:], np.nan)

    for i in range(regions_labels.shape[0]):
        for j in range(regions_labels.shape[1]):
            regions_labels[i, j] = stats.mode(labels[:, i, j])[0]

    geographic_plot(data=regions_labels,
                    lons_lats=lons_lats, levels=n_regions-1)

    return regions_labels, s_scores


def region_calculation(n_regions=4, show_silhouette=True):

    try:
        with np.load('av_model_dataYearly.npz') as av_m:
            av_matrix = av_m['matrix']
    except:
        av_matrix = generate_yearly_data()

    with np.load('lons_lats.npz') as ll:
        lons_lats = ll['lons_lats']

    region_labels, silhouette_scores = region_clusters(
        data=av_matrix, lons_lats=lons_lats, n_regions=n_regions, silhouette=show_silhouette)

    print('Silhouette Average: ', np.mean(silhouette_scores))
    if show_silhouette:
        plt.hist(silhouette_scores)
        plt.show()

    return region_labels


def average_by_region(matrix=None, chemical=0, r_labels=None, n_regions=4):

    data = np.full((matrix.shape[0], n_regions), np.nan)

    for i in range(matrix.shape[0]):
        d = [[0, 0]] * n_regions
        for j in range(r_labels.shape[0]):
            for k in range(r_labels.shape[1]):
                if not np.isnan(r_labels[j][k]):
                    d[int(r_labels[j, k])][0] += matrix[i, j, k, chemical]
                    d[int(r_labels[j, k])][1] += 1
        for region in range(n_regions):
            data[i,region] = d[region][0] / d[region][1]

    return data


def main():
    n_regions = 4

    try:
        with np.load('region_labels.npz') as r_labels:
            region_labels = r_labels['matrix']
    except:
        region_labels = region_calculation(
            n_regions=n_regions, show_silhouette=False)
        np.savez_compressed('region_labels.npz', matrix=region_labels)

    print('Fetching Data...')
    with np.load('av_model_data30.npz') as m:
        av_matrix = m['matrix']
    with open("datetimes.txt", "rb") as fp:
        dates = pickle.load(fp)
    print('Finished Fetching Data')

    mode = 'kmeans'
    n_clusters = 5

    data = []
    for i in range(4):
        data.append(average_by_region(matrix=av_matrix, chemical=i,
                                      r_labels=region_labels, n_regions=n_regions))
        print(np.amin(data[i]), np.amax(data[i]))
        # Clustering
        if mode == 'kmeans':
            clustered_data = clustering(
                data=data[i], n_clusters=n_clusters, mode='kmeans')
        elif mode == 'hierarchical':
            clustered_data = clustering(
                data=data[i], n_clusters=n_clusters, mode='hierarchical')

        print("The " + str(n_clusters) + " cluster sizes are:")
        cluster_sizes = [len(list(compress(data[i], clustered_data.labels_ == cluster)))
                         for cluster in range(n_clusters)]
        print(cluster_sizes)

        s_avg = silhouette_plot(labels=clustered_data.labels_, data=data[i],
                                plotGraph=True, n_clusters=n_clusters)

        # Keeping relevant dates
        new_d = []
        for i in range(len(dates)):
            if i % 30 == 0:
                new_d.append(dates[i])

        new_d.pop()

        timeseries_plot(data=clustered_data.labels_, t=new_d)


if __name__ == "__main__":
    main()
