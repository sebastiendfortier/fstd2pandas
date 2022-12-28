# -*- coding: utf-8 -*-
import logging
import math

import numpy as np
import pandas as pd

import rpnpy.librmn.all as rmn

from fstpy import DATYP_DICT
from fstpy.utils import to_numpy

from .dataframe import add_columns, add_ip_info_columns, reorder_columns
from .std_dec import convert_rmndate_to_datetime
from .std_vgrid import set_vertical_coordinate_type

class SelectError(Exception):
    pass


def select_meta(df: pd.DataFrame) -> pd.DataFrame:
    meta_df = df.loc[df.nomvar.isin(
        ["!!", "P0", "PT", ">>", "^^", "^>", "HY", "!!SF"])]
    return meta_df


def select_with_meta(df: pd.DataFrame, nomvar: list) -> pd.DataFrame:
    """Select fields with accompaning meta data  

    :param df: dataframe to select from  
    :type df: pd.DataFrame  
    :param nomvar: list of nomvars to select   
    :type nomvar: list  
    :raises SelectError: if dataframe is empty, if nothing to select or if variable not found in dataframe  
    :return: dataframe with selection results  
    :rtype: pd.DataFrame  
    """
    if df.empty:
        raise SelectError(f'dataframe is empty - nothing to select into')

    if not isinstance(nomvar,list):
        nomvar = [nomvar]
        
    results = []

    if len(nomvar) == 0:
        raise SelectError(f'nomvar is empty - nothing to select')

    for var in nomvar:
        res_df = df.loc[df.nomvar == var]
        if res_df.empty:
            raise SelectError(f'missing {var} in dataframe')
        results.append(res_df)

    meta_df = select_meta(df)

    if not meta_df.empty:
        results.append(meta_df)

    selection_result_df = pd.concat(results, ignore_index=True)

    selection_result_df = metadata_cleanup(selection_result_df)

    return selection_result_df


def metadata_cleanup(df: pd.DataFrame, strict_toctoc=True) -> pd.DataFrame:
    """Cleans the metadata from a dataframe according to rules.   

    :param df: dataframe to clean  
    :type df: pd.DataFrame  
    :return: dataframe with only cleaned meta_data  
    :rtype: pd.DataFrame  
    """

    if df.empty:
        return df
        
    df = set_vertical_coordinate_type(df)

    no_meta_df = df.loc[~df.nomvar.isin(["!!", "P0", "PT", ">>", "^^", "^>", "HY", "!!SF"])]

    # get deformation fields
    grid_deformation_fields_df = get_grid_deformation_fields(df, no_meta_df)

    sigma_ips = get_sigma_ips(no_meta_df)

    hybrid_ips = get_hybrid_ips(no_meta_df)

    # get P0's
    p0_fields_df = get_p0_fields(df, no_meta_df, hybrid_ips, sigma_ips)

    # get PT's
    pt_fields_df = get_pt_fields(df, no_meta_df, sigma_ips)

    # get HY
    hy_field_df = get_hy_field(df, hybrid_ips)

    pressure_ips = get_pressure_ips(no_meta_df)

    # get !!'s strict
    toctoc_fields_df = get_toctoc_fields(df, no_meta_df, hybrid_ips, sigma_ips, pressure_ips, strict_toctoc)

    new_df = pd.concat([grid_deformation_fields_df, p0_fields_df,
                    pt_fields_df, hy_field_df, toctoc_fields_df,no_meta_df], ignore_index=True)

    # new_df.sort_index(inplace=True)
    # new_df.reset_index(inplace=True)

    return new_df


class VoirError(Exception):
    pass


def voir(df: pd.DataFrame, style=False):
    """Displays the metadata of the supplied records in the rpn voir format"""
    if df.empty:
        raise VoirError('No records to process')

    to_print_df = df.copy()
    to_print_df['datyp'] = to_print_df['datyp'].map(DATYP_DICT)
    to_print_df['datev'] = to_print_df['datev'].apply(convert_rmndate_to_datetime)
    to_print_df['dateo'] = to_print_df['dateo'].apply(convert_rmndate_to_datetime)
    to_print_df = add_ip_info_columns(to_print_df)

    res_df = to_print_df.sort_values(by=['nomvar', 'level'], ascending=[True, False])

    if style:
        res_df = res_df.drop(columns=['dateo', 'grid', 'run', 'implementation', 'ensemble_member', 'd', 'ip1_kind', 'ip2_dec', 'ip2_kind', 'ip2_pkind',
                                      'ip3_dec', 'ip3_kind', 'ip3_pkind', 'date_of_observation', 'date_of_validity', 'forecast_hour', 'd', 'surface', 'follow_topography', 'ascending', 'interval','label','unit','description','zapped','filtered','interpolated','unit_converted','bounded','missing_data','ensemble_extra_info','vctype','data_type_str','level','ip1_pkind','multiple_modifications'], errors='ignore')
        res_df = reorder_columns(res_df, ordered=['nomvar', 'typvar', 'etiket', 'ni', 'nj', 'nk', 'datev', 'level',
                                                  ' ', 'ip1', 'ip2', 'ip3', 'deet', 'npas', 'datyp', 'nbits', 'grtyp', 'ig1', 'ig2', 'ig3', 'ig4'])
    else:
        res_df = res_df.drop(columns=['datev', 'grid', 'run', 'implementation', 'ensemble_member', 'd', 'ip1_kind', 'ip2_dec', 'ip2_kind', 'ip2_pkind', 'path', 'key', 'shape',
                                      'ip3_dec', 'ip3_kind', 'ip3_pkind', 'date_of_observation', 'date_of_validity', 'forecast_hour', 'd', 'surface', 'follow_topography', 'ascending', 'interval','label','unit','description','zapped','filtered','interpolated','unit_converted','bounded','missing_data','ensemble_extra_info','vctype','data_type_str','level','ip1_pkind','multiple_modifications'], errors='ignore')

    #print('    NOMV TV   ETIQUETTE        NI      NJ    NK (DATE-O  h m s) FORECASTHOUR      IP1        LEVEL        IP2       IP3     DEET     NPAS  DTY   G   IG1   IG2   IG3   IG4')
    print('\n%s' % res_df.reset_index(drop=True).to_string(header=True))


class FstStatError(Exception):
    pass


def fststat(df: pd.DataFrame):
    """Produces summary statistics for a dataframe

    :param df: input dataframe
    :type df: pd.DataFrame
    """
    logging.info('fststat')
    
    if df.empty:
        raise FstStatError('fststat - no records to process')
    df = add_columns(df,['ip_info'])    
    compute_stats(df)

def compute_stats(df: pd.DataFrame):
    pd.options.display.float_format = '{:8.6E}'.format
    df['min'] = None
    df['max'] = None
    df['mean'] = None
    df['std'] = None
    df['min_pos'] = None
    df['max_pos'] = None
    # print(f"        {'nomvar':6s} {'typvar':6s} {'level':8s} {'ip1':9s} {'ip2':4s} {'ip3':4s} {'dateo':10s} {'etiket':14s} {'mean':8s} {'std':8s} {'min_pos':12s} {'min':8s} {'max_pos':12s} {'max':8s}")
    # i  = 0
    for row in df.itertuples():
        d = to_numpy(row.d)
        min_pos = np.unravel_index(np.argmin(d), (row.ni, row.nj))
        df.at[row.Index,'min_pos'] = (min_pos[0] + 1, min_pos[1]+1)
        max_pos = np.unravel_index(np.argmax(d), (row.ni, row.nj))
        df.at[row.Index,'max_pos'] = (max_pos[0] + 1, max_pos[1]+1)
        df.at[row.Index,'min'] = np.min(d)
        df.at[row.Index,'max'] = np.max(d)
        df.at[row.Index,'mean'] = np.mean(d)
        df.at[row.Index,'std'] = np.std(d)
        # print(f'{i:5d} - {row.nomvar:6s} {row.typvar:6s} {row.level:8.6f} {row.ip1:9d} {row.ip2:4d} {row.ip3:4d} {row.dateo:10d} {row.etiket:14s} {np.mean(d):8.6f} {np.std(d):8.6f} {str(min_pos):12s} {np.min(d):8.6f} {str(max_pos):12s} {np.max(d):8.6f}')
        # i = i+1
    print(df[['nomvar','typvar','level','ip1','ip2','ip3','dateo','etiket','mean','std','min_pos','min','max_pos','max']].to_string())    
    


def get_kinds_and_ip1(df: pd.DataFrame) -> dict:
    ip1s = df.ip1.unique()
    kinds = {}
    for ip1 in ip1s:
        if math.isnan(ip1):
            continue
        (_, kind) = rmn.convertIp(rmn.CONVIP_DECODE, int(ip1))
        if kind not in kinds.keys():
            kinds[kind] = []
        kinds[kind].append(ip1)

    return kinds


def get_ips(df: pd.DataFrame, sigma=False, hybrid=False, pressure=False) -> list:
    kinds = get_kinds_and_ip1(df)

    ip1_list = []
    if sigma:
        if 1 in kinds.keys():
            ip1_list.extend(kinds[1])
    if hybrid:
        if 5 in kinds.keys():
            ip1_list.extend(kinds[5])
    if pressure:
        if 2 in kinds.keys():
            ip1_list.extend(kinds[2])
    return ip1_list


def get_model_ips(df: pd.DataFrame) -> list:
    return get_ips(df, sigma=True, hybrid=True)


def get_sigma_ips(df: pd.DataFrame) -> list:
    return get_ips(df, sigma=True)


def get_pressure_ips(df: pd.DataFrame) -> list:
    return get_ips(df, pressure=True)


def get_hybrid_ips(df: pd.DataFrame) -> list:
    return get_ips(df, hybrid=True)


def get_toctoc_fields(df: pd.DataFrame, no_meta_df:pd.DataFrame, hybrid_ips: list, sigma_ips: list, pressure_ips: list, strict=True):

    toctoc_df = df.loc[df.nomvar=='!!']
    
    df_list = []

    hybrid_fields_df = pd.DataFrame(dtype=object)
    # hybrid
    if len(hybrid_ips):
        hybrid_fields_df = no_meta_df.loc[no_meta_df.ip1.isin(hybrid_ips)]

    hybrid_grids = []
    if not hybrid_fields_df.empty:
        hybrid_grids = hybrid_fields_df.grid.unique()

    # sigma
    sigma_fields_df = pd.DataFrame(dtype=object)
    if len(sigma_ips):
        sigma_fields_df = no_meta_df.loc[no_meta_df.ip1.isin(sigma_ips)]

    sigma_grids = []
    if not sigma_fields_df.empty:
        sigma_grids = sigma_fields_df.grid.unique()

    # pressure
    pressure_fields_df = pd.DataFrame(dtype=object)
    if len(pressure_ips):
        pressure_fields_df = no_meta_df.loc[no_meta_df.ip1.isin(pressure_ips)]

    pressure_grids = []
    if not pressure_fields_df.empty:
        pressure_grids = pressure_fields_df.grid.unique()

    for grid in hybrid_grids:
        # grids_no_meta_df = no_meta_df.loc[no_meta_df.grid == grid]
        # vctypes = list(grids_no_meta_df.vctype.unique())
        hyb_toctoc_df = toctoc_df.loc[(toctoc_df.grid == grid) & (
            toctoc_df.ig1.isin([1003, 5001, 5002, 5003, 5004, 5005, 5100, 5999, 21001, 21002]))]
        # vctypes = list(hyb_toctoc_df.ig1.unique())
        # vctypes = numeric_vctype_to_string(vctypes)
        if not hyb_toctoc_df.empty:
            df_list.append(hyb_toctoc_df)

    # vcode 1001 -> Sigma levels
    # vcode 1002 -> Eta levels
    for grid in sigma_grids:
        # grids_no_meta_df = no_meta_df.loc[no_meta_df.grid == grid]
        sigma_toctoc_df = toctoc_df.loc[(toctoc_df.grid == grid) & (toctoc_df.ig1.isin([1001, 1002]))]
        # vctypes = list(sigma_toctoc_df.ig1.unique())
        # vctypes = numeric_vctype_to_string(vctypes)
        if not sigma_toctoc_df.empty:
            df_list.append(sigma_toctoc_df)

    # vcode 2001 -> Pressure levels
    for grid in pressure_grids:
        presure_toctoc_df = toctoc_df.loc[(df.grid == grid) & (df.ig1 == 2001)]
        if not presure_toctoc_df.empty:
            df_list.append(presure_toctoc_df)

    toctoc_fields_df = pd.DataFrame(dtype=object)

    if len(df_list):
        toctoc_fields_df = pd.concat(df_list, ignore_index=True)

    toctoc_fields_df = toctoc_fields_df.drop_duplicates(subset=['grtyp', 'nomvar', 'typvar', 'ni', 'nj', 'nk', 'ip1',
                                                                'ip2', 'ip3', 'deet', 'npas', 'nbits', 'ig1', 'ig2', 'ig3', 'ig4', 'datev', 'dateo', 'datyp'], ignore_index=True)

    # toctoc_fields_df.sort_index(inplace=True)
    return toctoc_fields_df

# def numeric_vctype_to_string(vctypes):
#     vctype_list = []
#     for vctype in vctypes:
#         if vctype == 5002:
#             vctype_list.append('HYBRID_5002')
#         elif vctype == 5001:
#             vctype_list.append('HYBRID_5001')
#         elif vctype == 5005:
#             vctype_list.append('HYBRID_5005')
#         elif vctype == 2001:
#             vctype_list.append('PRESSURE_2001')
#         elif vctype == 1002:
#             vctype_list.append('ETA_1002')
#         elif vctype == 1001:
#             vctype_list.append('SIGMA_1001')
#         else:
#             vctype_list.append('UNKNOWN')
#     return vctype_list        


def get_hy_field(df: pd.DataFrame, hybrid_ips: list):

    hy_field_df = pd.DataFrame(dtype=object)
    if len(hybrid_ips):
        hy_field_df = df.loc[df.nomvar == "HY"]

    hy_field_df = hy_field_df.drop_duplicates(subset=['grtyp', 'nomvar', 'typvar', 'ni', 'nj', 'nk', 'ip1', 'ip2',
                                                      'ip3', 'deet', 'npas', 'nbits', 'ig1', 'ig2', 'ig3', 'ig4', 'datev', 'dateo', 'datyp'], ignore_index=True)
    # hy_field_df.sort_index(inplace=True)

    return hy_field_df


def get_grid_deformation_fields(df: pd.DataFrame, no_meta_df: pd.DataFrame):
    col_subset = ['nomvar', 'typvar', 'etiket', 'ni', 'nj', 'nk', 'dateo', 'ip1', 'ip2', 'ip3', 'deet', 'npas', 'ig1', 'ig2', 'ig3', 'ig4']

    grid_deformation_fields_df = pd.DataFrame(dtype=object)

    groups = no_meta_df.groupby(['grid', 'dateo', 'datev', 'deet', 'npas'])

    df_list = []

    for (grid, dateo, _, deet, npas), group in groups:
        if len(list(group.ni.unique())) > 1:
            logging.error(f'grid with fields of different sizes for ni {group.ni.unique()}')
        if len(list(group.nj.unique())) > 1:
            logging.error(f'grid with fields of different sizes for nj {group.nj.unique()}')    

        lat_df    = get_specific_meta_field(df, col_subset, "^^", grid, dateo, deet, npas)
        lon_df    = get_specific_meta_field(df, col_subset, ">>", grid, dateo, deet, npas)
        tictac_df = get_specific_meta_field(df, col_subset, "^>", grid, dateo, deet, npas)

        df_list.append(lat_df)
        df_list.append(lon_df)
        df_list.append(tictac_df)

    if len(df_list):
        grid_deformation_fields_df = pd.concat(df_list, ignore_index=True)

    grid_deformation_fields_df = grid_deformation_fields_df.drop_duplicates(subset=col_subset, ignore_index=True)

    # grid_deformation_fields_df.sort_index(inplace=True)

    return grid_deformation_fields_df

def get_specific_meta_field(df, col_subset, nomvar, grid, dateo, deet, npas):
    subset = col_subset.copy()
    # try very strict match
    field_df = df.loc[(df.nomvar == nomvar) & (df.grid == grid) & (df.dateo == dateo) & (df.deet == deet) & (df.npas == npas)] 

    if field_df.empty:
        # try a strict match
        field_df = df.loc[(df.nomvar == nomvar) & (df.grid == grid) & (df.dateo == dateo)]
        if field_df.empty:
            # try a loose match
            field_df = df.loc[(df.nomvar == nomvar) & (df.grid == grid)]
            if not field_df.empty:
                # we found something on loose match - remove the duplicates        
                subset.remove('deet')
                subset.remove('npas')
                subset.remove('dateo')
                field_df = field_df.drop_duplicates(subset = subset)
        else:    
            # we found something on strict match - remove the duplicates    
            subset.remove('deet')
            subset.remove('npas')
            field_df = field_df.drop_duplicates(subset = subset)
    else:
        # we found something on very strict match - remove the duplicates
        field_df = field_df.drop_duplicates(subset = subset)
    return field_df    

def get_p0_fields(df: pd.DataFrame, no_meta_df: pd.DataFrame, hybrid_ips: list, sigma_ips: list):

    p0_df = df.loc[df.nomvar=='P0']

    p0_fields_df = pd.DataFrame(dtype=object)

    hybrid_grids = set()
    for ip1 in hybrid_ips:
        hybrid_grids.add(no_meta_df.loc[no_meta_df.ip1 == ip1].iloc[0]['grid'])

   
    df_list = []
    for grid in hybrid_grids:
        ni = no_meta_df.loc[no_meta_df.grid == grid].ni.unique()[0]
        nj = no_meta_df.loc[no_meta_df.grid == grid].nj.unique()[0]
        df_list.append(p0_df.loc[(p0_df.grid == grid) & (p0_df.ni == ni) & (p0_df.nj == nj)])

    if len(df_list):
        p0_fields_df = pd.concat(df_list, ignore_index=True)


    sigma_grids = set()
    for ip1 in sigma_ips:
        sigma_grids.add(no_meta_df.loc[no_meta_df.ip1 == ip1].iloc[0]['grid'])

    df_list = []
    for grid in sigma_grids:
        ni = no_meta_df.loc[no_meta_df.grid == grid].ni.unique()[0]
        nj = no_meta_df.loc[no_meta_df.grid == grid].nj.unique()[0]
        df_list.append(p0_df.loc[(p0_df.grid == grid) & (p0_df.ni == ni) & (p0_df.nj == nj)])

    if len(df_list):
        p0_fields_df = pd.concat(df_list, ignore_index=True)

    p0_fields_df.drop_duplicates(subset=['grtyp', 'nomvar', 'typvar', 'ni', 'nj', 'nk', 'ip1', 'ip2', 'ip3', 'deet',
                                         'npas', 'nbits', 'ig1', 'ig2', 'ig3', 'ig4', 'datev', 'dateo', 'datyp'], inplace=True, ignore_index=True)

    # p0_fields_df.sort_index(inplace=True)

    return p0_fields_df


def get_pt_fields(df: pd.DataFrame, no_meta_df: pd.DataFrame, sigma_ips: list):
    pt_df = df.loc[df.nomvar=='PT']
    
    pt_fields_df = pd.DataFrame(dtype=object)
    
    sigma_grids = set()
    for ip1 in sigma_ips:
        sigma_grids.add(no_meta_df.loc[no_meta_df.ip1 == ip1].iloc[0]['grid'])

    df_list = []
    for grid in list(sigma_grids):
        ni = no_meta_df.loc[no_meta_df.grid == grid].ni.unique()[0]
        nj = no_meta_df.loc[no_meta_df.grid == grid].nj.unique()[0]
        df_list.append(pt_df.loc[(pt_df.grid == grid) & (pt_df.ni == ni) & (pt_df.nj == nj)])

    if len(df_list):
        pt_fields_df = pd.concat(df_list, ignore_index=True)

    pt_fields_df.drop_duplicates(subset=['grtyp', 'nomvar', 'typvar', 'ni', 'nj', 'nk', 'ip1', 'ip2', 'ip3', 'deet',
                                         'npas', 'nbits', 'ig1', 'ig2', 'ig3', 'ig4', 'datev', 'dateo', 'datyp'], inplace=True, ignore_index=True)

    # pt_fields_df.sort_index(inplace=True)

    return pt_fields_df
