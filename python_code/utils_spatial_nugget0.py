# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 13:50:49 2021

@author: saulg
"""

import numpy as np
import os
import pickle
import gstools as gs  # "conda install -c conda-forge gstools"
import matplotlib.pyplot as plt
import netCDF4
import shapely.geometry
from shapely.geometry import MultiPolygon, shape
from shapely.ops import cascaded_union, unary_union
import fiona
from copy import deepcopy

class krigging_interpolation():
    # Establish where files are located. If location doesn't exist, it will 
    # create a new location.
    
    def __init__(self, data_root ='./Datasets', figures_root = './Figures Spatial'):
        # Data Root is the location where data will loaded from. Saved to class
        # in order to reference later in other functions.
        if os.path.isdir(data_root) is False:
            os.makedirs(data_root)
        self.data_root = data_root
        
        # Fquifer Root is the location to save figures.
        if os.path.isdir(figures_root) is False:
            os.makedirs(figures_root)
        self.figures_root = figures_root


    def read_pickle(self, file, root):
        # Opens generic pickle file based on file path and loads data.
        file = root + file + '.pickle'
        with open(file, 'rb') as handle:
            data = pickle.load(handle)
        return data


    def Save_Pickle(self, Data, name:str, protocol:int = 3):
        # Saves generic pickle file based on file path and loads data.
        with open(self.Data_root + '/' + name + '.pickle', 'wb') as handle:
            pickle.dump(Data, handle, protocol= protocol)


    def Shape_Boundary(self, shape_file_path):
        # Load shapefile boundary
        self.shape_file_path = shape_file_path
        user_shape = fiona.open(shape_file_path)
        return user_shape
    
    
    def extract_dataframe_data(self, well_data, skip_month):
        # Extract a regular interval row of data
        self.data_min = well_data.stack(level=0).min()
        self.data_max = well_data.stack(level=0).max()
        well_data = well_data.iloc[::skip_month,:]
        return well_data
        
    
    def create_grid_polygon(self, polygon, x_cells:int = None, y_cells:int = None, res = 0.1, plot=True):

        # create grid coordinates for kriging, make x and y steps the same
        # x_steps is the number of cells in the x-direction


        # Unpack shapfile boundary fiona uses south east corner (sec) and 
        # north west corner (nwc) to determine boundary
        polygon_boundary = polygon.bounds
        sec_lon = polygon_boundary[0]
        sec_lat = polygon_boundary[1]
        nwc_lon = polygon_boundary[2]       
        nwc_lat = polygon_boundary[3]
        
        
        # Determine length of aquifer used in variaogram as well as setting up grid
        self.poly_lon_len = abs(polygon_boundary[2] - polygon_boundary[0])
        self.poly_lat_len = abs(polygon_boundary[3] - polygon_boundary[1])
        if plot:
            print(f'Longitude range is: {self.poly_lon_len}.')
            print(f'Latitude range is: {self.poly_lat_len}.')


        # Extent of grid
        if x_cells is not None:  self.res = float(self.poly_lon_len/x_cells)
        elif y_cells is not None: self.res = float(self.poly_lat_len/y_cells) 
        elif res is not None: self.res = res
        else: self.res = 0.1
        if plot:
            print(f'Grid Resolution is {self.res}.')

        grid_lat = np.arange(nwc_lat, sec_lat, -self.res).tolist()
        grid_long = np.arange(sec_lon, nwc_lon, self.res).tolist()
        
        
        # Mask Creation: Create an array the shape of grid. 1 will be that the cell
        # is located inside shape. 0 is outside shape.
        mask_array = np.ones((len(grid_lat), len(grid_long))) # Array creation
        polygon_object = self._select_shape(polygon)


        # Loop through every point to see if point is in shape
        for i, lat in enumerate(grid_lat):
            for j, long in enumerate(grid_long):
                point_temp = shapely.geometry.Point(long, lat) 
                if not polygon_object.contains(point_temp): mask_array[i,j] = 0
        # Save mask to class to use in interpolation. Change 0s to NANs to make
        # Data visualization correct.
        self.mask_array = np.where(mask_array == 0, np.nan, 1)
        if plot:
            plt.imshow(self.mask_array)
            plt.title('Aquifer Boundary')
            plt.show()
        return grid_long, grid_lat
    

    def _select_shape(self, polygon):
        shapes = []
        for i, poly in enumerate(polygon):
            if poly['geometry']['type'] == 'Polygon': shapes.append(shape(poly['geometry']))
            elif poly['geometry']['type'] == 'MultiPolygon': shapes += list(MultiPolygon(shape(poly['geometry'])))
        boundary = unary_union(shapes)
        return boundary


    def netcdf_setup(self, grid_long, grid_lat, timestamp, filename):
        # setup a netcdf file to store the time series of rasters
        # copied from other lab code - you probably don't need everything here
        
        filepath = self.data_root + '/' + filename
        if os.path.exists(filepath):
            os.remove(filepath)
            
        file = netCDF4.Dataset(self.data_root + '/' + filename, 'w', format="NETCDF4")
        
        lon_len  = len(grid_long)  # size of grid in x_dir
        lat_len  = len(grid_lat)  # size of grid in y_dir
        
        time = file.createDimension("time", None) # time dimension - can extend e.g., size=0
        lat  = file.createDimension("lat", lat_len)  # create lat dimension in netcdf file of len lat_len
        lon  = file.createDimension("lon", lon_len)  # create lon dimension in netcdf file of len lon_len
        
        time      = file.createVariable("time", np.float64, ("time"))
        latitude  = file.createVariable("lat", np.float64, ("lat"))  # create latitude varilable
        longitude = file.createVariable("lon", np.float64, ("lon")) # create longitude varilbe
        tsvalue   = file.createVariable("tsvalue", np.float64, ('time', 'lat', 'lon'), fill_value=-9999)
        
        # Netcdf seems to flip lat/long for building grid
        latitude[:] = grid_lat[:] 
        longitude[:] = grid_long[:] 

    
        latitude.long_name = "Latitude"
        latitude.units = "degrees_north"
        latitude.axis = "Y"
        
        longitude.long_name = "Longitude"
        longitude.units = "degrees_east"
        longitude.axis = "X"
        
        timestamp = list(timestamp.to_pydatetime())
        units = 'days since 0001-01-01 00:00:00'
        calendar = 'standard'
        time[:] = netCDF4.date2num(dates = timestamp, units = units, calendar= calendar)
        time.axis = "T"
        time.units = units
        
        return file, tsvalue
    
    
    def fit_model_var(self, x_c, y_c, values, influence = 0.125, plot=True):
        # the current version specifies a vargiogram rather than fitting one
        
        bin_num = 20  # number of bins in the experimental variogram
        
        # first get the coords and determine distances
        x_delta = self.poly_lon_len # distance across x coords
        y_delta = self.poly_lat_len  # distance across y coords
        max_dist = np.sqrt(x_delta**2 + y_delta**2) # Hyp. of grid
        influence_distance = max_dist * influence #distance wells are correlated
        # setup bins for the variogram
        bins_c = np.linspace(0, max_dist, bin_num)  # bin edges in variogram, bin_num of bins
    
        # compute the experimental variogram
        bin_cent_c, gamma_c = gs.vario_estimate_unstructured((x_c, y_c), values, bins_c)
        # bin_center_c is the "lag" of the bin, gamma_c is the value
    
        data_var = np.var(values)
        data_std = np.std(values)
        fit_var = gs.Stable(dim=2, var=data_var, len_scale=influence_distance, nugget=data_std)
        # the code commented out above "fits" the variogram to the actual data
        # here we just specifiy the range, with the sill equal to the
        # variation of the data, and the nugget equal to the standard deviation.
        # the nugget is probably too big using this approach
        # we could set the nugget to 0
    
        if plot:
            # plot the variogram to show fit and print out variogram paramters
            ax1 = fit_var.plot(x_max=max_dist)  # plot model variogram
            ax1.plot(bin_cent_c, gamma_c)  # plot experimental variogram
            plt.show()
            print(fit_var)  # print out model variogram parameters.
        return fit_var
    
    def krig_field(self, var_fitted, x_c, y_c, values, grid_x, grid_y, date, plot=True):
        # use GSTools to krig  the well data, need coords and value for each well
        # use model variogram paramters generated by GSTools
        # fast - is faster the variogram fitting
        krig_map = gs.krige.Ordinary(var_fitted, cond_pos=[x_c, y_c], cond_val=values)
        krig_map.structured([grid_x, grid_y]) # krig_map.field is the numpy array of values
        krig_map.field = krig_map.field.T * self.mask_array
        if plot==True:
            plt.pcolor(grid_x, grid_y, krig_map.field, cmap = 'gist_rainbow', vmin=self.data_min, vmax=self.data_max)
            plt.colorbar()
            plt.scatter(x_c, y_c, c='r')
            plt.title('Groundwater Surface: ' + str(date.strftime('%Y-%m-%d')))
            plt.savefig(self.figures_root  + '/' + str(date.strftime('%Y-%m-%d')+'_01'))
            plt.show()
            
            plt.pcolor(grid_x, grid_y, krig_map.field, cmap = 'Spectral')
            plt.colorbar()
            plt.scatter(x_c, y_c, c='r')
            plt.title('Batch Groundwater Surface: ' + str(date.strftime('%Y-%m-%d')))
            plt.savefig(self.figures_root  + '/' + str(date.strftime('%Y-%m-%d')+'_02'))
            plt.show()
        return krig_map
    
