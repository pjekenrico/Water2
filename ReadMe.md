# Data

## Data source:
Copernicus Marine Environment Monitoring Service

Link to product:
http://marine.copernicus.eu/services-portfolio/access-to-products/?option=com_csw&view=details&product_id=NORTHWESTSHELF_REANALYSIS_BIO_004_011

ORIGINAL FILE FORMAT:	NetCDF-4

TEMPORAL COVERAGE:	from 1998-01-01T00:00:00Z to 2018-12-30T00:00:00Z

EMPORAL RESOLUTION:	daily-mean

## Data description:
dataset-PHOS-model-daily --> phosphate concentration (model)
dataset-NITR-model-daily --> nitrate concentration (model)
dataset-DOXY-model-daily --> dissolved oxygen concentration (model)
dataset-CHL-model-daily --> chlorophyll-a concentration (model)

dataset-CHL-satellite-daily --> chlorophyll-a concentration (satellite)
dataset-SPM-satellite-monthly --> suspended sediment concentration (satellite)


# Code
## clustering.py
Different types of clustering algorithms are implemented plus a few helper functions.
Exemple of how to use them is also given.

## double_clustering.py
Implementation of the region calculations, plus an example application.

## NetCDF_basic.py
Original code provided, used for data exploration.

## read_satellite_data.py
Code used for the analysis of the satellite data.

## ripser.py
Code used for topological data analysis (TDA)

## save_data.py
Code used for saving the provided data in a numpy matrix format.

## validation.py
Different methods for the validation of clusters are implemented (silhouette scores, elbow method, dendograms).

## visualization.py
Different visulization functions for different purposes are implemented.
Also a class for creating and saving timplapses anymations is implemented
