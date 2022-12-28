# -*- coding: utf-8 -*-
import copy
import ctypes
import logging
import math
import numpy as np
import pandas as pd
from typing import Tuple
import rpnpy.librmn.proto as pt
from .dataframe import add_path_and_key_columns
import rpnpy.librmn.all as rmn

def get_df_from_grid(grid_params: dict) -> pd.DataFrame:
    """For grid types Z, Y or U, produces a DataFrame of >>,^^ or ^> fields

    :param grid_params: horizontal grid definition dictionnary
    :type grid_params: dict
    :return: DataFrame of >>,^^ or ^> fields
    :rtype: pd.DataFrame
    """
    g = grid_params
    if (g['grtyp'] == 'Z') or (g['grtyp'] == 'Y'):
        meta_df = pd.DataFrame(
            [
                {'nomvar': '>>', 'typvar': 'X', 'etiket': '', 'ni': g['ni'], 'nj':1, 'nk':1, 'dateo':0, 'ip1':g['ig1'], 
                'ip2':g['ig2'], 'ip3':0, 'deet':0, 'npas':0, 'datyp':5, 'nbits':32, 'grtyp':g['grref'], 
                'ig1':g['ig1ref'], 'ig2':g['ig2ref'], 'ig3':g['ig3ref'], 'ig4':g['ig4ref'], 'datev':0, 'd':g['ax']},
                {'nomvar': '^^', 'typvar': 'X', 'etiket': '', 'ni': 1, 'nj': g['nj'], 'nk':1, 'dateo':0, 'ip1':g['ig1'], 
                'ip2':g['ig2'], 'ip3':0, 'deet':0, 'npas':0, 'datyp':5, 'nbits':32, 'grtyp':g['grref'], 
                'ig1':g['ig1ref'], 'ig2':g['ig2ref'], 'ig3':g['ig3ref'], 'ig4':g['ig4ref'], 'datev':0, 'd':g['ay']}
            ]
            )

    elif g['grtyp'] == 'U':
        meta_df = pd.DataFrame(
            [
                {'nomvar': '^>', 'typvar': 'X', 'etiket': '', 'ni': g['axy'].shape[0], 'nj':1, 'nk':1, 'dateo':0, 
                'ip1':g['ig1'], 'ip2':g['ig2'], 'ip3':0, 'deet':0, 'npas':0, 'datyp':5, 'nbits':32, 'grtyp':g['grref'], 
                'ig1':g['ig1ref'], 'ig2':g['ig2ref'], 'ig3':g['ig3ref'], 'ig4':g['ig4ref'], 'datev':0, 'd':g['axy']}
            ]
            )
    else:
        meta_df = pd.DataFrame(dtype=object)
    return meta_df



class GridDefinitionError(Exception):
    pass

def get_grid_definition_params(df):
    if df.empty:
        raise GridDefinitionError('Empty DataFrame!')

    if 'path' not in df.columns:
        df = add_path_and_key_columns(df)

    if df.path.unique().size > 1:
        raise GridDefinitionError('More than one path in DataFrame!')

    if df.grid.unique().size > 1:
        raise GridDefinitionError('More than one grid in DataFrame!')

    no_meta_df = df.loc[~df.nomvar.isin(["^^", ">>", "^>", "!!", "!!SF", "HY", "P0", "PT"])]
    if no_meta_df.grtyp.unique().size > 1:
        raise GridDefinitionError('More than one grtyp in DataFrame!')

    grtyp = no_meta_df.grtyp.unique()[0]
    ni = no_meta_df.ni.unique()[0]
    nj = no_meta_df.nj.unique()[0]

    grid_id = -1

    if ( (grtyp == "A") or (grtyp == "B") or (grtyp == "E") or
        (grtyp == "G") or (grtyp == "L") or (grtyp == "N") or
        (grtyp == "S") or (grtyp == "U") or (grtyp == "X") or
        (grtyp == "Y") or (grtyp == "Z") or (grtyp == "#")):

        if ( (grtyp == "Y") or (grtyp == "Z")  or (grtyp == "#") or (grtyp == "U")):
            file_id = rmn.fstopenall(df.path.unique()[0], rmn.FST_RO)
            try:
                rec = rmn.fstprm(int(no_meta_df.iloc[0].key))
            except rmn.FSTDError:
                rmn.fstcloseall(file_id)
                raise GridDefinitionError('Error while getting record data!')
            try:
                grid_id = rmn.readGrid(file_id, rec)
            except rmn.EzscintError:
                rmn.fstcloseall(file_id)
                raise GridDefinitionError('Error while reading grid!')
            rmn.fstcloseall(file_id)
        else:
            info_s = no_meta_df.iloc[0]
            ig1 = info_s.ig1
            ig2 = info_s.ig2
            ig3 = info_s.ig3
            ig4 = info_s.ig4
            cgrtyp = ctypes.c_char_p(grtyp.encode('utf-8'))
            grid_id = pt.c_ezqkdef(ni, nj, cgrtyp, ig1, ig2, ig3, ig4, 0)
            try:
                grid_id = rmn.decodeGrid(grid_id)
            except rmn.RMNError:
                grid_id = rmn.ezgxprm(grid_id)


    else:
        logging.error(f"{grtyp} grid type is not supported")

    return grid_id

class GetSubGridsError(Exception):
    pass

def get_subgrids(grid_params: dict):
    keys = list(grid_params.keys())
    if 'subgrid' not in keys:
        raise GetSubGridsError('No subgrids found!')
    else:    
        if len(grid_params['subgrid']) != 2:
            raise GetSubGridsError('For U type grid, there should only be 2 subgrids!')
        gd1 = grid_params['subgrid'][0]
        gd2 = grid_params['subgrid'][1]
        return gd1, gd2

class Get2DLatLonError(Exception):
    pass

def get_2d_lat_lon_arr(grid_params: dict) -> 'list(Tuple(np.ndarray,np.ndarray))':
    if not isinstance(grid_params, dict):
        raise Get2DLatLonError('grid_id must be a valid grid definition as type dict')

    keys = list(grid_params.keys())
    if 'subgrid' in keys:
        if len(grid_params['subgrid']) != 2:
            raise Get2DLatLonError('For U type grid, there should only be 2 subgrids!')
        gd1 = grid_params['subgrid'][0]
        gd2 = grid_params['subgrid'][1]
        lat1 = rmn.gdll(gd1)['lat']
        lon1 = rmn.gdll(gd1)['lon']
        lat2 = rmn.gdll(gd2)['lat']
        lon2 = rmn.gdll(gd2)['lon']
        lats = np.concatenate([lat1, lat2], axis=1)
        lons = np.concatenate([lon1, lon2], axis=1)
        latlons = (lats,lons)

    else:
        g = rmn.gdll(grid_params)
        latlons = (g['lat'], g['lon'])
    return latlons


def get_2d_lat_lon_df(df: pd.DataFrame) -> pd.DataFrame:
    """Gets the latitudes and longitudes as 2d arrays associated with the supplied grids

    :return: a pandas Dataframe object containing the lat and lon meta data of the grids
    :rtype: pd.DataFrame
    :raises Get2DLatLonError: no records to process
    """
    if df.empty:
        raise Get2DLatLonError('Empty DataFrame!')

    if 'path' not in df.columns:
        df = add_path_and_key_columns(df)

    path_groups = df.groupby('path')
    for _, path_df in path_groups:
        
        grid_groups = path_df.groupby(['grid'])
        for _, grid_df in grid_groups:
            no_meta_df = grid_df.loc[~grid_df.nomvar.isin(["^^", ">>", "^>", "!!", "!!SF", "HY", "P0", "PT"])].reset_index(drop=True)

            if no_meta_df.empty:
                continue

            tictic_df = pd.DataFrame([no_meta_df.iloc[0].to_dict()])
            tactac_df = pd.DataFrame([no_meta_df.iloc[0].to_dict()])

            df_list = []
            grtyp_groups = no_meta_df.groupby('grtyp')
            for grtyp, grtyp_df in grtyp_groups:
                if (grtyp == 'X'):
                    logging.warning(f"{grtyp} is an unsupported grid type!")
                    continue

                grid_params = get_grid_definition_params(grtyp_df)
                (lat, lon) = get_2d_lat_lon_arr(grid_params)

                tictic_df['nomvar'] = 'LA'
                tictic_df['d'] = [lat]
                tictic_df['ni'] = lat.shape[0]
                tictic_df['nj'] = lat.shape[1]

                tactac_df['nomvar'] = 'LO'
                tactac_df['d'] = [lon]
                tactac_df['ni'] = lon.shape[0]
                tactac_df['nj'] = lon.shape[1]


                df_list.append(tictic_df)
                df_list.append(tactac_df)

    latlon_df = pd.concat(df_list, ignore_index=True)
    return latlon_df

class GlobalGridError(Exception):
    pass

def is_global_grid(grid_params: dict, lon: np.ndarray, epsilon: float = 0.001) -> 'Tuple(bool,bool)':
    """Checks with the information received if the grid is a global grid as well as if the first longitude is repeated or not

    :param grid_params: grid parameters obtained from get_grid_definition_params
    :type grid_params: dict
    :param lon: 2d fortran order longitude matrix obtained with get_2d_lat_lon_arr
    :type lon: np.ndarray
    :param epsilon: Epsilon value for comparison operators, defaults to 0.001
    :type epsilon: float, optional
    :return: is a global grid, has  longitude repetition 
    :rtype: (bool, bool)
    """

    global_grid = False
    repetition = False

    grtyp = grid_params['grtyp']

    # Grilles de type A et B sont globales - pas besoin de faire de verification
    if (grtyp == "A") or (grtyp == "G"):
        global_grid = True

    elif (grtyp == "B"):
        global_grid = True
        repetition = True

    else:
        lon_data = lon.flatten(order='F')

        if (grtyp == "Z"):
            # Est-ce que la longitude se repete?
            # Si oui, grille globale avec point qui se repete
            # Si non, ce n'est pas une grille globale
            if (_equal(lon_data[0], lon_data[-1], epsilon)):
                global_grid = True
                repetition = True
        elif (grtyp == "L"):
            ni = grid_params['ni']
            dlon = grid_params['dlon']

            if (math.fmod(360.0, dlon) != 0):
                nb_points = ni * dlon

                # On couvre plus que 360 deg et longitude[dernier point] <= (nb_points+dlon)-360
                if ((_greater_or_equal(nb_points, 360.0, epsilon)) and (_lower_or_equal(lon_data[-1], ((nb_points + dlon) - 360.0), epsilon)) ):
                    # Cas 2 : on fait le tour MAIS la valeur du point qui se repete est differente du point 0
                    logging.warning("Global grid with the first longitude repeated at the end of the grid but with a different longitude!  will be treated as a non global grid!")
                    repetition = True
                elif ((_greater_or_equal(nb_points, 360.0, epsilon)) and (_lower_than(lon_data[-1], 360.0, epsilon))):
                    # Cas 3 : on fait le tour -- dernier point ne se repete pas - pas de distance egale entre le dernier point et 0 degre 
                    global_grid = True
                    repetition = False

            else:
                # Globale avec point qui ne se repete pas
                if (_equal((ni * dlon), 360.0, epsilon)):
                    global_grid = True
                # Globale avec point qui se repete
                elif (_equal((ni * dlon), (360.0 + dlon), epsilon)):
                    global_grid = True
                    repetition = True

    return global_grid, repetition

def _equal(value, threshold, epsilon=0.00001):
    return (math.fabs(value - threshold) <= epsilon)

def _greater_or_equal(value, threshold, epsilon=0.00001):
    return ((value > threshold) or _equal(value, threshold, epsilon))

def _lower_than(value, threshold, epsilon=0.00001):
    return (not _greater_or_equal(value, threshold, epsilon))

def _lower_or_equal(value, threshold, epsilon=0.00001):
    return (value < threshold or _equal(value, threshold, epsilon))

# def get_df_from_vgrid(vgrid_descriptor: vgd.VGridDescriptor, ip1: int, ip2: int) -> pd.DataFrame:
#     v = vgrid_descriptor
#     vcoord = vertical_coord_to_dict(v)
#     vers = str(vcoord['VERSION']).zfill(3)
#     ig1 = int(''.join([str(vcoord['KIND']),vers]))
#     data = vcoord['VTBL']
#     meta_df = pd.DataFrame([{'nomvar':'!!', 'typvar':'X', 'etiket':'', 'ni':data.shape[0], 'nj':data.shape[1], 'nk':1, 'dateo':0, 'ip1':ip1, 'ip2':ip2, 'ip3':0, 'deet':0, 'npas':0, 'datyp':5, 'nbits':64, 'grtyp':'X', 'ig1':ig1, 'ig2':0, 'ig3':0, 'ig4':0, 'datev':0, 'd':data}])
#     return meta_df

# def vertical_coord_to_dict(vgrid_descriptor) -> dict:
#     myvgd = vgrid_descriptor
#     vcoord={}
#     vcoord['KIND'] = vgd.vgd_get(myvgd,'KIND')
#     vcoord['VERSION'] = vgd.vgd_get(myvgd,'VERSION')
#     vcoord['VTBL'] = np.asfortranarray(np.squeeze(vgd.vgd_get(myvgd,'VTBL')))
#     return vcoord

# gp = {
#     'grtyp' : 'Z',
#     'grref' : 'E',
#     'ni'    : 90,
#     'nj'    : 45,
#     'lat0'  : 35.,
#     'lon0'  : 250.,
#     'dlat'  : 0.5,
#     'dlon'  : 0.5,
#     'xlat1' : 0.,
#     'xlon1' : 180.,
#     'xlat2' : 1.,
#     'xlon2' : 270.
# }
# g = rmn.encodeGrid(gp)

# 'xlat1': 0.0,
# 'xlon1': 180.0,
# 'xlat2': 1.0,
# 'xlon2': 270.0,
# 'ni': 90,
# 'nj': 45,
# 'rlat0': 34.059606166461926,
# 'rlon0': 250.23401123256826,
# 'dlat': 0.5,
# 'dlon': 0.5,
# 'lat0': 35.0,
# 'lon0': 250.0,
# 'grtyp': 'Z',
# 'grref': 'E',
# 'ig1ref': 900,          ig1
# 'ig2ref': 10,           ig2
# 'ig3ref': 43200,        ig3
# 'ig4ref': 43200,        ig4
# 'ig1': 66848,
# 'ig2': 39563,
# 'ig3': 0,
# 'ig4': 0,
# 'id': 0,
# 'tag1': 66848,
# 'tag2': 39563,
# 'tag3': 0,
# 'shape': (90, 45)}
#   nomvar typvar etiket  ni  nj  nk  dateo    ip1    ip2  ip3  ...  datyp  nbits  grtyp  ig1 ig2    ig3    ig4  datev        grid
# 0     >>      X         90   1   1      0  66848  39563    0  ...      5     32      E  900  10  43200  43200      0  6684839563
# 1     ^^      X          1  45   1      0  66848  39563    0  ...      5     32      E  900  10  43200  43200      0  6684839563

# {'nomvar':'>>', typvar:'X', 'etiket':'', 'ni':g.ni, nj:1, 'nk':1, 'dateo':0, 'ip1':g.ig1, 'ip2':g.ig2, 'ip3':0, 'datyp':5, 'nbits':32, 'grtyp':g.grref, 'ig1':g.ig1ref, 'ig2':g.ig2ref, 'ig3':g.ig3ref, 'ig4'g.ig4ref:, 'datev':0, 'd':g.ax}
# {'nomvar':'^^', typvar:'X', 'etiket':'', 'ni':1, nj:g.nj, 'nk':1, 'dateo':0, 'ip1':g.ig1, 'ip2':g.ig2, 'ip3':0, 'datyp':5, 'nbits':32, 'grtyp':g.grref, 'ig1':g.ig1ref, 'ig2':g.ig2ref, 'ig3':g.ig3ref, 'ig4'g.ig4ref:, 'datev':0, 'd':g.ay}
