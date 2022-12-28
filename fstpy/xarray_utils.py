# -*- coding: utf-8 -*-

from fstpy.dataframe import add_columns, add_shape_column
from fstpy.std_io import get_field_dtype, get_lat_lon
import xarray as xr
import pandas as pd
import numpy as np
import dask.array as da

def convert_to_cmc_xarray(df, timeseries=False, attributes=False):
    """creates a xarray from the provided data

    :param timeseries: if True, organizes the xarray into a time series, defaults to False
    :type timeseries: bool, optional
    :param attributes: if True, will add attributes to the data arrays, defaults to False
    :type attributes: bool, optional
    :return: xarray containing contents of cmc standard files
    :rtype: xarray.DataSet
    """

    # dask_config_set(**{'array.slicing.split_large_chunks': True})
    df = add_columns(df)
    df = add_shape_column(df)
    counter = 0
    data_list = []
    grid_groups = df.groupby(df.grid)

    for _,grid_df in grid_groups:
        positional_df = grid_df.loc[grid_df.nomvar.isin(['>>','^^'])]
        
        counter += 1
        if len(grid_groups.size()) > 1:
            lat_name = f'rlat{counter}' if not positional_df.empty else f'y{counter}'
            lon_name = f'rlon{counter}' if not positional_df.empty else f'x{counter}'
            datev_name = f'time{counter}'
            level_name = f'level{counter}'
        else:
            lat_name = 'rlat' if not positional_df.empty else 'y'
            lon_name = 'rlon' if not positional_df.empty else 'x'
            datev_name = 'time'
            level_name = 'level'
        lat_lon_df = get_lat_lon(grid_df)
        grid_shape = grid_df.loc[~grid_df.nomvar.isin(['!!','>>','^^','^>','HY','P0','PT'])].iloc[0]['shape']
        longitudes = get_longitude_data_array(lat_lon_df,lon_name,grid_shape)
        #print(longitudes.shape)
        latitudes = get_latitude_data_array(lat_lon_df,lat_name,grid_shape)
        #print(latitudes.shape)
        nomvar_groups = grid_df.groupby(['nomvar','ip1_kind'])
        for _,nomvar_df in nomvar_groups:
            # nomvar_df.sort_values(by='level',inplace=True)
            nomvar_df.sort_values(by='level',ascending=nomvar_df.ascending.unique()[0],inplace=True)
            nomvar_df = nomvar_df.reset_index(drop=True)
            if nomvar_df.iloc[0]['nomvar'] in ['!!','>>','^^','^>','HY']:
                continue
            if len(nomvar_df.datev.unique()) > 1 and timeseries:
                time_dim = get_date_of_validity_data_array(nomvar_df,datev_name)
                nomvar_df.sort_values(by=['date_of_validity'],ascending=True,inplace=True)
            else: #nomvar_df.ip1.unique() == 1:
                level_dim = get_level_data_array(nomvar_df,level_name,nomvar_df.ascending.unique()[0])
                # nomvar_df.sort_values(by=['level'],ascending=nomvar_df.ascending.unique()[0],inplace=True)
            attribs = {}
            if attributes:
                attribs = set_data_array_attributes(attribs,nomvar_df, timeseries)

            nomvar = nomvar_df.iloc[-1]['nomvar']
            if timeseries:
                data_list.append(get_variable_data_array(nomvar_df, nomvar, attribs, time_dim, datev_name, latitudes, lat_name, longitudes, lon_name, timeseries=True))
            else:
                data_list.append(get_variable_data_array(nomvar_df, nomvar, attribs, level_dim, level_name, latitudes, lat_name, longitudes, lon_name, timeseries=False))

    d = {}
    for variable in data_list:
            d.update({variable.name:variable})

    ds = xr.Dataset(d)

    return ds


def set_data_array_attributes(attribs:dict, nomvar_df:pd.DataFrame) -> dict:
    """Sets the data array attributes

    :param attribs: dictionnary of attribute to attach to data arrays
    :type attribs: dict
    :param nomvar_df: dataframe organized by nomvar
    :type nomvar_df: pd.DataFrame
    :return: filled dict of atributes
    :rtype: dict
    """

    attribs = nomvar_df.iloc[-1].to_dict()
    attribs = remove_keys(attribs,['key','nomvar','etiket','ni','nj','nk','shape','ig1','ig2','ig3','ig4','ip1','ip2','ip3','datyp','dateo','pkind','datev','grid','d'])
    for k,v in attribs.items():
        attribs = set_attrib(nomvar_df,attribs,k)
    # attribs = set_attrib(nomvar_df,attribs,'ip1_kind')
    # attribs = set_attrib(nomvar_df,attribs,'surface')
    # attribs = set_attrib(nomvar_df,attribs,'date_of_observation')
    # attribs = set_attrib(nomvar_df,attribs,'path')
    # attribs = set_attrib(nomvar_df,attribs,'ensemble_member')
    # attribs = set_attrib(nomvar_df,attribs,'implementation')
    # attribs = set_attrib(nomvar_df,attribs,'run')
    # attribs = set_attrib(nomvar_df,attribs,'label')
    #if not timeseries:
    #    attribs = set_attrib(nomvar_df,attribs,'date_of_validity')

    #attribs = remove_keys(nomvar_df,attribs,['ni','nj','nk','shape','ig1','ig2','ig3','ig4','ip1','ip2','ip3','datyp','dateo','pkind','datev','grid','fstinl_params','d','file_modification_time','ensemble_member','implementation','run','label'])
    
    return attribs


def remove_keys(a_dict,keys):
    for k in keys:
        a_dict.pop(k,None)
    return a_dict    

def set_attrib(nomvar_df,attribs,key):
    attribs[key] = np.array(getattr(nomvar_df,key).to_list()) if len(getattr(nomvar_df,key).unique()) > 1 else attribs[key]
    return attribs

def get_date_of_validity_data_array(df,date_of_validity_name):
    times = df['date_of_validity'].to_numpy()
    time = xr.DataArray(
        times,
        dims=[date_of_validity_name],
        coords=dict(time = times),
        name = date_of_validity_name
        )
    return time

def get_level_data_array(df,level_name,ascending):
    levels = df['level'].to_numpy(dtype='float32')
    level = xr.DataArray(
        levels,
        dims=[level_name],
        coords={level_name:levels},
        name = level_name
        )
    return level

def get_latitude_data_array(lat_lon_df,lat_name,shape):
    if not lat_lon_df.empty:
        lati = lat_lon_df.query('nomvar=="^^"').iloc[0]['d'].flatten()
    else:    
        lati = np.arange(0,shape[1],dtype=np.int32)
    attribs = {
        'long_name' : 'latitude in rotated pole grid',
        'standard_name' : 'grid_latitude',
        'units' : 'degrees',
        'axis' : 'Y',
    }
    lat = xr.DataArray(
        lati,
        dims=[lat_name],
        coords={lat_name:lati},
        name=lat_name,
        attrs=attribs
        )
    return lat
    
def get_longitude_data_array(lat_lon_df,lon_name,shape):

    if not lat_lon_df.empty:
        # loni = np.flip(lat_lon_df.query('nomvar==">>"').iloc[0]['d'].flatten(),axis=0)
        # loni = (loni-163.41278)*-1
        loni = lat_lon_df.query('nomvar==">>"').iloc[0]['d'].flatten()
    else:
        loni = np.arange(0,shape[0],dtype=np.int32) 

    attribs = {
        'long_name' : 'longitude in rotated pole grid',
        'standard_name' : 'grid_longitude',
        'units' : 'degrees',
        'axis' : 'X'
    }
    lon = xr.DataArray(
        loni,
        dims=[lon_name],
        coords={lon_name:loni},
        name=lon_name,
        attrs=attribs
        )
       
    return lon

def get_variable_data_array(df, name, attribs, dim, dim_name, latitudes, lat_name, longitudes, lon_name,timeseries=False):
    field_dtype = get_field_dtype(df.iloc[0]['datyp'],df.iloc[0]['nbits'])    
    values = da.stack(df['d'].to_list())

    if not timeseries:
        dimensions = [dim_name,lon_name,lat_name]
        coordinates = {dim_name:dim,lat_name:latitudes,lon_name:longitudes}
    else:
        dimensions = [dim_name,lon_name,lat_name]
        coordinates = {dim_name:dim,lat_name:latitudes,lon_name:longitudes}

    arr_da = xr.DataArray(
        values,
        dims=dimensions,
        coords=coordinates,
        name=name,
        attrs=attribs
        )    
    return arr_da.astype(field_dtype)     
