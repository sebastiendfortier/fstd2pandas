# -*- coding: utf-8 -*-
import copy
import itertools
import multiprocessing as mp
import os
from pathlib import Path
from . import FSTPY_PROGRESS

try:
    from tqdm import tqdm
except ModuleNotFoundError as e:
    FSTPY_PROGRESS = False
    
import pandas as pd


from fstpy.xarray_utils import convert_to_cmc_xarray

from .utils import initializer, to_numpy


class StandardFileReaderError(Exception):
    pass


class StandardFileReader:
    """Class to handle fst files. Opens, reads the contents of an fst file or files into a pandas dataframe and closes. Extra metadata columns are added to the dataframe if specified.    

        :param filenames: path to file or list of paths to files  
        :type filenames: str|pathlib.Path|list[str], does not accept wildcards (numpy has 
                         many tools for this)  
        :param decode_metadata: adds extra columns, defaults to False  
            'unit':str, unit name   
            'unit_converted':bool  
            'description':str, field description   
            'date_of_observation':datetime, of the date of observation   
            'date_of_validity': datetime, of the date of validity   
            'level':float32, decoded ip1 level   
            'ip1_kind':int32, decoded ip1 kind   
            'ip1_pkind':str, string repr of ip1_kind int   
            'data_type_str':str, string repr of data type   
            'label':str, label derived from etiket   
            'run':str, run derived from etiket   
            'implementation': str, implementation derived from etiket   
            'ensemble_member': str, ensemble member derived from etiket   
            'surface':bool, True if the level is a surface level   
            'follow_topography':bool, indicates if this type of level follows topography   
            'ascending':bool, indicates if this type of level is in ascending order   
            'vctype':str, vertical level type   
            'forecast_hour': timedelta, forecast hour obtained from deet * npas / 3600   
            'ip2_dec':value of decoded ip2    
            'ip2_kind':kind of decoded ip2    
            'ip2_pkind':printable kind of decoded ip2   
            'ip3_dec':value of decoded ip3   
            'ip3_kind':kind of decoded ip3   
            'ip3_pkind':printable kind of decoded ip3   
        :type decode_metadata: bool, optional  
        :param query: parameter to pass to dataframe.query method, to select specific records  
        :type query: str, optional  
    """
    meta_data = ["^>", ">>", "^^", "!!", "!!SF", "HY", "P0", "PT", "E1"]

    @initializer
    def __init__(self, filenames, decode_metadata=False, query=None):
        """init instance"""
        if isinstance(self.filenames, Path):
            self.filenames = str(self.filenames.absolute())
        elif isinstance(self.filenames, str):
            self.filenames = os.path.abspath(str(self.filenames))
        elif isinstance(self.filenames, list):
            self.filenames = [os.path.abspath(str(f)) for f in filenames]
        else:
            raise StandardFileReaderError('Filenames must be str or list\n')

    def to_pandas(self) -> pd.DataFrame:
        from .std_io import get_dataframe_from_file
        from .dataframe import add_columns, drop_duplicates
        """creates the dataframe from the provided file metadata

        :return: df
        :rtype: pd.Dataframe
        """

        if isinstance(self.filenames, list):
            # if len(self.filenames) < 100:
            df_list = []
            for f in tqdm(self.filenames, desc = 'Reading files') if FSTPY_PROGRESS else self.filenames:
                df = get_dataframe_from_file(f, self.query)
                df_list.append(df)
            df = pd.concat(df_list,ignore_index=True)    
            # else:    
            #     # convert to list of tuple (path,query)
            #     self.filenames = list(zip(self.filenames, itertools.repeat(self.query)))

            #     df = parallel_get_dataframe_from_file(
            #         self.filenames,  get_dataframe_from_file, n_cores=min(mp.cpu_count(), len(self.filenames)))

        else:
            df = get_dataframe_from_file(self.filenames, self.query)

        if self.decode_metadata:
            df = add_columns(df)

        df = drop_duplicates(df)

        return df
     
    def to_cmc_xarray(self):
        df = self.to_pandas()
        return convert_to_cmc_xarray(df)

def to_cmc_xarray(df):
    return convert_to_cmc_xarray(df)
    

def compute(df: pd.DataFrame,remove_path_and_key:bool=True) -> pd.DataFrame:
    """Converts all dask arrays contained in the 'd' column, by numpy arrays

    :param df: input DataFrame
    :type df: pd.DataFrame
    :param remove_path_and_key: remove path and key column after conversion, defaults to True
    :type remove_path_and_key: bool, optional
    :return: modified dataframe with numpy arrays instead of dask arrays
    :rtype: pd.DataFrame
    """
    from .dataframe import add_path_and_key_columns
    import dask as da
    new_df = copy.deepcopy(df)
    
    new_df = add_path_and_key_columns(new_df)
    
    no_path_df = new_df.loc[new_df.path.isna()]

    groups = new_df.groupby('path')
    
    df_list = []
    
    if not no_path_df.empty:
        for row in no_path_df.itertuples():
             no_path_df.at[row.Index, 'd'] = to_numpy(row.d)
        # d = da.compute(*list(no_path_df['d'].values))
        # for i,row in enumerate(no_path_df.itertuples()):
        #      no_path_df.at[row.Index, 'd'] = d[i]
        df_list.append(no_path_df)
        
    for _, current_df in groups:

        current_df = current_df.sort_values('key')

        for row in current_df.itertuples():
             current_df.at[row.Index, 'd'] = to_numpy(row.d)
        # d = da.compute(*list(current_df['d'].values))
        # for i,row in enumerate(current_df.itertuples()):
        #      current_df.at[row.Index, 'd'] = d[i]

        df_list.append(current_df)
    
    new_df = pd.concat(df_list).sort_index()
    
    if remove_path_and_key:
        new_df = new_df.drop(['path','key'], axis=1, errors='ignore')
    
    return new_df
