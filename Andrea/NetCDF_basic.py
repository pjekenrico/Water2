# -*- coding: utf-8 -*-
"""
@author: Lorinc Meszaros
"""
#==============================================================================
import netCDF4
from netCDF4 import Dataset
import os
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib 
import matplotlib.pyplot as plt
from math import pi
from numpy import cos,sin
from scipy.spatial import cKDTree
import datetime as dt

#==============================================================================

sub = "CHL" 
nc_path = "C:/Users/andre/OneDrive/Desktop/TUD/Mathematical Data Science/Project/Data_WQ/MetO-NWS-BIO-dm-"+ sub+ ".nc"
dataset = Dataset(nc_path)


sub_key = list(dataset.variables)[1] #For CHL and PHOS this should be 1 instead of 0

#==============================================================================  
timestep=10; #choose timestep

fh = Dataset(nc_path, mode='r')
time = fh.variables['time']
jd = netCDF4.num2date(time[:],time.units)

d = []
for dd in jd:
    d.append(dt.date(dd.year,dd.month,dd.day))


lons = fh.variables['longitude'][:]
lats = fh.variables['latitude'][:]

lons, lats = np.meshgrid(lons,lats)

Sub_variable = fh.variables[sub_key][:]
Sub_variable = np.squeeze(Sub_variable)
fh.close()

#==============================================================================
#PLOT
# Get some parameters for the Stereographic Projection

Sub_plot=Sub_variable[timestep,:,:]

#Plot
matplotlib.rcParams['figure.figsize'] = (10,10) 

proj=ccrs.Mercator()
m = plt.axes(projection=proj)
# Put a background image on for nice sea rendering.
m.stock_img()
m.coastlines(resolution='110m')
m.add_feature(cfeature.BORDERS)
gl=m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
#Plot data
plt.contourf(lons, lats, Sub_plot, 50,
             transform=ccrs.PlateCarree())

# Add Colorbar
cbar = plt.colorbar()
cbar.set_label(dataset.variables[sub_key].units)

# Add Title
plt.title(sub + ' concentration')

plt.show()
#==============================================================================
#Spatial subset loop

#region coordinates
ylat_north = 53.8
ylat_south = 52.8
xlon_east = 8
xlon_west = 4.2

sub_sub=np.full([int(Sub_variable.shape[1]), int(Sub_variable.shape[2])], np.nan)
lat_sub=np.full([int(Sub_variable.shape[1]), int(Sub_variable.shape[2])], np.nan)
lon_sub=np.full([int(Sub_variable.shape[1]), int(Sub_variable.shape[2])], np.nan)
for j in range(0,int(Sub_variable.shape[2])):
    for i in range(0,int(Sub_variable.shape[1])):
        #if the element by element lat and lons lie within the lat and lons specified for the region, then the indices for each point are saved in idxi and idxj, while the actual data itself is written to the 'domainrun' matrix (previously entirely filled with NaNs). This results in a matrix for the region, containing only the data for that specific subdomain along with NaNs everywhere else
        
        if lats[i,j]<=ylat_north and lats[i,j]>=ylat_south and lons[i,j]>=xlon_west and lons[i,j]<=xlon_east:
            #lat_sub and lon_sub contain the actual lat and lons for the subregion         
            lat_sub[i,j]=lats[i,j]
            lon_sub[i,j]=lons[i,j]
            sub_sub[i,j]=Sub_variable[timestep,i,j]
        else:
            pass
#==============================================================================
#PLOT SUBSET
#Plot
matplotlib.rcParams['figure.figsize'] = (10,10) 
            
proj=ccrs.Mercator()
m = plt.axes(projection=proj)
# Put a background image on for nice sea rendering.
m.stock_img()
m.coastlines(resolution='110m')
m.add_feature(cfeature.BORDERS)
gl=m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False

#Plot data
plt.contourf(lon_sub, lat_sub, sub_sub[:,:], 50,
             transform=ccrs.PlateCarree())

# Add Colorbar
cbar = plt.colorbar()
cbar.set_label(dataset.variables[sub_key].units)

# Add Title
plt.title(sub + ' concentration')

plt.show()

#==============================================================================
# Get time series at given location
loni = 4.75027 
lati = 52.983622

def kdtree_fast (lats, lons, lat_0, lon_0):
        rad_factor= pi/180.0 #for trignometry, need angles in radians
        # Read Lat,Long from file to numpy arrays
        latvals= lats[:]*rad_factor
        lonvals= lons[:]*rad_factor
        ny,nx = latvals.shape
        clat,clon = cos(latvals),cos(lonvals)
        slat,slon = sin(latvals),sin(lonvals)
        # Build kd-tree from big arrays of 3D coordinates
        triples = list(zip(np.ravel(clat*clon), np.ravel(clat*slon), np.ravel(slat)))
        kdt = cKDTree(triples)
        lat0_rad = lat_0 * rad_factor
        lon0_rad = lon_0 * rad_factor
        clat0,clon0 = cos(lat0_rad),cos(lon0_rad)
        slat0,slon0 = sin(lat0_rad),sin(lon0_rad)
        dist_sq_min, minindex_1d = kdt.query([clat0*clon0, clat0*slon0, slat0])
        iy_min, ix_min = np.unravel_index(minindex_1d, latvals.shape)
        return iy_min,ix_min


#Option2: if lon/lat is 2D array
#Looking up array indices using KD-Tree 


iy,ix = kdtree_fast(lats, lons, lati, loni)
print ('Exact Location lat-lon:', [lati,loni])
print ('Closest lat-lon:', lats[iy,ix], lons[iy,ix])
print ('Array indices [iy,ix]=', iy, ix)    
                
#Get all time records of variable [vname] at indices [iy,ix]
h = Sub_variable[:,ix,iy]

#Plot ime series
plt.figure(figsize=(16,4))
plt.plot_date(d,h)
plt.grid()
plt.ylabel(dataset.variables[sub_key].units)
plt.title('%s at Lon=%.2f, Lat=%.2f' % (sub, lons[iy, ix], lats[iy, ix]))

plt.show()
