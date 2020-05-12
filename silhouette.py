import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from itertools import compress
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram

def silhouette_plot(labels=None, data=None, name_model='', plotGraph=False, n_clusters=0):
    '''Returns the silhouette metric and the respective graph (if required):
    Receives three parameters:
        - labels: Labels of clustering model
        - data: data where the model is applied
        - plotGraph (default False)
        - name_model = Name of the evaluated model

    s_avg  : Average silhouette metric for the clustering model
    '''

    # n_clusters = len(set(labels.tolist()))
    if n_clusters == 0:
        print('Error: 0 clusters')
        return 0
    elif n_clusters == 1:
        return 1

    s_samples = silhouette_samples(X=data, labels=labels)
    s_avg = silhouette_score(X=data, labels=labels)

    if plotGraph:
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)
        ax.set_xlim([-1, 1])
        ax.set_ylim([0, len(data) + (n_clusters + 1) * 10])
        #ax.set_ylim([0, data.getnnz() + (n_clusters + 1)])

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            # ith_cluster_silhouette_values = s_samples[labels == i]
            ith_cluster_silhouette_values = np.array(list(compress(s_samples, labels == i)))
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]

            y_upper = y_lower + size_cluster_i
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                             0, ith_cluster_silhouette_values,
                             facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        # Adjust the plot
        ax.set_title("Silhouette plot for the various clusters for the model " +
                     name_model + ".\nAverage value: {}".format(s_avg))
        ax.set_xlabel("Silhouette coefficient values")
        ax.set_ylabel("Cluster label")
        ax.axvline(x=s_avg, color="red", linestyle="--")
        ax.set_yticks([])  # Clear the yaxis labels / ticks
        ax.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.show()

    return s_avg


def elbowPlot(inertiaVals, n_cluster):

    if isinstance(inertiaVals,list):
        inertia = np.array(inertiaVals)
    else:
        inertia = inertiaVals

    derivative2 = inertia[2:] -2*inertia[1:-1] + inertia[0:-2]

    fig, ax = plt.subplots(1,2)
    ax[0].plot(n_cluster,inertiaVals)
    ax[1].plot(n_cluster[:-2],derivative2)

    ax[0].set_title("Inertia Elbow method")
    ax[1].set_title("Derivative Elbow method")

    ax[0].grid('on')
    ax[1].grid('on')

    ax[0].set_xlabel('$n_{C}$')
    ax[0].set_ylabel('$\sum^{n_{C}}_{i = 1}\sum^{n}_{j=1} min_{x_i \in C_i} ||x_j - \mu_i||^2$')
    plt.show()



def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

    plt.title('Hierarchical Clustering Dendrogram')
    # plot the top three levels of the dendrogram
    plt.xlabel("Number of points in cluster")
    plt.axhline(y=0.25, color="tab:orange", linestyle="--")
    plt.axhline(y=0.4, color="tab:orange", linestyle="--")
    plt.show()