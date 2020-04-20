import sklearn.cluster as cluster
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import compress
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def clustering(data=None, n_clusters=10, mode="kmeans"):

    print("Starting the Clustering Procedure, using mode: " + mode)

    if(mode == 'kmeans'):
        clusterer = cluster.KMeans(n_clusters=n_clusters, random_state=0,
                                   init='k-means++', n_init=30, max_iter=500, tol=1)
        clusterer.fit(data)

    print("Finished Clustering.")
    return clusterer

# Load data
matrix = np.load("model_data.npy")
lons_lats = np.load("lons_lats.npy")

# Select timestep
timestep = 2000

# Straighten-out data for clustering
data = matrix[timestep, :, :, :]
straight_data = np.array([[0, 0, 0, 0, 0, 0]])

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if min(data[i, j, :]) > -1000:
            d = np.append(data[i, j, :], [i, j])
            straight_data = np.append(straight_data, [d], axis=0)

straight_data = straight_data[1:]

# Clustering
n_clusters = 4
clustered_data = clustering(data=straight_data[:, 0:4], n_clusters=n_clusters)
print("The " + str(n_clusters) + " cluster sizes:")
print([len(list(compress(straight_data, clustered_data.labels_ == i)))
       for i in range(n_clusters)])

# Saving lables in a spatial martix
labels = np.full(lons_lats.shape[:-1], np.nan)

for i in range(len(straight_data)):
    labels[int(straight_data[i, 4]), int(straight_data[i, 5])] = clustered_data.labels_[i]

# Plotting the clusters
matplotlib.rcParams['figure.figsize'] = (10, 10)
proj = ccrs.Mercator()
m = plt.axes(projection=proj)

# Put a background image on for nice sea rendering.
m.stock_img()
m.coastlines(resolution='10m')
m.add_feature(cfeature.BORDERS)
gl = m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False

# Plot data
plt.contourf(lons_lats[:, :, 0], lons_lats[:, :, 1], labels, 50,
             transform=ccrs.PlateCarree())

# Add Colorbar
cbar = plt.colorbar()

# Add Title
plt.title('Time: ' + str(timestep) + ' clusters')

plt.show()
