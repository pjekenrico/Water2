import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from itertools import compress
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram

def silhouette_plot(labels=None, data=None, name_model='', plotGraph=False, n_clusters=0):
    '''Returns the silhouette metric and the respective graph (if required):
    
    labels:     Labels of clustering model
    data:       data where the model is applied
    plotGraph:  (default False)
    name_model: Name of the evaluated model
    n_clusters: number of clusters

    s_avg  : Average silhouette metric for the clustering model
    '''
    # Handling of special numbers of clusters
    if n_clusters == 0:
        print('Error: 0 clusters')
        return 0
    elif n_clusters == 1:
        return 1

    # Calculations
    s_samples = silhouette_samples(X=data, labels=labels)
    s_avg = silhouette_score(X=data, labels=labels)

    # Plotting of the silhouettes
    if plotGraph:
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)
        ax.set_xlim([-1, 1])
        ax.set_ylim([0, len(data) + (n_clusters + 1) * 10])

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
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
    '''
    Plots the values of the inertis computed by clustering, to be analysed through the elbow method
    '''

    if isinstance(inertiaVals,list):
        inertia = np.array(inertiaVals)
    else:
        inertia = inertiaVals

    derivative2 = inertia[2:] -2*inertia[1:-1] + inertia[0:-2]

    fig = plt.figure(figsize = (8,5))
    plt.plot(n_cluster,inertiaVals, label = '$J$')
    plt.plot(n_cluster[1:-1], derivative2, label = r'$\ddot{J}$')
    plt.title("Elbow method", fontdict=dict(color="black", size=14))
    plt.xlabel('$n_{C}$', fontdict=dict(color="black", size=14))
    plt.xlim([np.min(n_cluster),np.max([n_cluster])])
    plt.xticks([2, 4, 6, 8, 10, 12, 14, 16, 18],\
        ["2","4","6","8","10","12","14","16","18"])
    plt.yticks([])
    plt.legend()
    plt.show()



def plot_dendrogram(model, **kwargs):
    '''
    Plots the dendogram obtained by hierarchical clustering
    '''

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
    plt.figure(figsize = (12,8))
    ax = plt.axes()

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix,leaf_rotation=0, **kwargs, ax = ax)

    plt.title('Hierarchical Clustering Dendrogram', fontdict=dict(color="black", size=14))
    plt.xlabel("Number of points in cluster", fontdict=dict(color="black", size=14))
    plt.ylabel("$\epsilon$", fontdict=dict(color="black", size=16))
    plt.axhline(y=0.23, color="tab:orange", linestyle="--", label = 'Feasible region')
    plt.axhline(y=0.4, color="tab:orange", linestyle="--")
    plt.grid('on', which = 'minor', axis = 'y', color='gray', linestyle='--', linewidth=0.7, alpha = 0.5)
    plt.legend()
    plt.show()