# -*- coding: utf-8 -*-
import copy
import logging
import math
import os
from pathlib import Path
from fstpy import FSTPY_PROGRESS

try:
    from tqdm import tqdm
except ModuleNotFoundError as e:
    FSTPY_PROGRESS = False

import numpy as np
import pandas as pd
import rpnpy.librmn.all as rmn

from fstpy.dataframe import add_path_and_key_columns
from fstpy.std_reader import compute

from .dataframe_utils import metadata_cleanup
from .std_io import get_field_dtype
from .utils import get_num_rows_for_reading, initializer, to_numpy


class StandardFileWriterError(Exception):
    pass


class StandardFileWriter:
    """Writes a standard file Dataframe to file. If no metada fields like ^^ and >> are found,
    an attempt will be made to load them from the original file so that they can be added to the output if not already present

    :param filename: path of file to write to
    :type filename: str
    :param df: dataframe to write
    :type df: pd.DataFrame
    :param mode: In 'dump' mode, no processing will be done on the dataframe 
                before writing, data must be present in the dataframe (df = compute(df)).
                If set to 'update', path must be an existing file. Only the 
                field metadata will be updated, the data itself will not be 
                modified. In 'write' mode, the data will be loaded, metadata 
                fields like '>>' will be added if not present default 'write'
    :type mode: str
    :param no_meta: if true these fields ["^>", ">>", "^^", "!!", "!!SF", "HY", "P0", "PT", "E1"] will be removed from the dataframe
    :type no_meta: bool
    :param overwrite: if True and dataframe inputfile is the same as the output file, records will be added to the file, defaults to False
    :type overwrite: bool, optional
    :param rewrite: overrides default rewrite value for fstecr, default None
    :type rewrite: bool, optional
    """
    modes = ['write', 'update', 'dump']

    @initializer
    def __init__(self, filename: str or Path, df: pd.DataFrame, mode='write', no_meta=False, overwrite=False, rewrite=None):
        self.validate_input()

    def validate_input(self):
        if self.df.empty:
            raise StandardFileWriterError(
                'StandardFileWriter - no records to process')

        if self.mode not in self.modes:
            raise StandardFileWriterError(
                f'StandardFileWriter - mode must have one of these values {self.modes}, you entered {self.mode}')

        self.filename = os.path.abspath(str(self.filename))
        self.file_exists = os.path.exists(self.filename)

        if self.file_exists and self.overwrite == False:
            raise StandardFileWriterError(
                'StandardFileWriter - file exists, use overwrite flag to avoid this error')

    def to_fst(self):
        """In write mode, gets the metadata fields if not already present and adds them to the dataframe.
        If not in update only mode, loads the actual data, opens the file writes the dataframe and closes.
        """
        # remove meta
        if self.no_meta:
            self.df = self.df.loc[~self.df.nomvar.isin(
                ["^>", ">>", "^^", "!!", "!!SF", "HY", "P0", "PT", "E1"])]

        if self.mode == 'dump':
            self._dump()
        elif self.mode == 'update':
            self._update()
        else:
            self._write()

    def _dump(self):
        if not(self.rewrite is None):
            rewrite = self.rewrite
        else:
            rewrite = True
        file_id = rmn.fstopenall(self.filename, rmn.FST_RW)
        for row in self.df.itertuples():
            rmn.fstecr(file_id, data=np.asfortranarray(to_numpy(row.d)), meta=self.df.loc[row.Index].to_dict(), rewrite=rewrite)
        rmn.fstcloseall(file_id)

    def _update(self):
        self.overwrite = True
        if not self.file_exists:
            raise StandardFileWriterError(
                'StandardFileWriter - file does not exist, cant update records')

        new_df = copy.deepcopy(self.df)
        
        new_df = add_path_and_key_columns(new_df)
        
        path = new_df.path.unique()
        
        
        if len(path) != 1:
            raise StandardFileWriterError(
                'StandardFileWriter - more than one path, cant update records')

        if path[0] != self.filename:
            raise StandardFileWriterError(
                'StandardFileWriter - path in dataframe is different from destination file path, cant update records')

        file_id = rmn.fstopenall(self.filename, rmn.FST_RW)
        for row in new_df.itertuples():
            rmn.fst_edit_dir(int(row.key), dateo=int(row.dateo), deet=int(row.deet), npas=int(row.npas), ni=int(row.ni), nj=int(row.nj), nk=int(row.nk), datyp=int(row.datyp), ip1=int(row.ip1), ip2=int(
                row.ip2), ip3=int(row.ip3), typvar=row.typvar, nomvar=row.nomvar, etiket=row.etiket, grtyp=row.grtyp, ig1=int(row.ig1), ig2=int(row.ig2), ig3=int(row.ig3), ig4=int(row.ig4), keep_dateo=False)
        rmn.fstcloseall(file_id)

    def _write(self):
        from fstpy.dataframe import add_path_and_key_columns

        self.df = metadata_cleanup(self.df)
        self.df = add_path_and_key_columns(self.df)
        self.df = self.df.sort_values(by=['path','key'])

        if self.rewrite is None:
            rewrite = set_rewrite(self.df)
        else:
            rewrite = self.rewrite

        num_rows = get_num_rows_for_reading(self.df)
        
        df_list = np.array_split(self.df, math.ceil(len(self.df.index)/num_rows))  # of records per block
         
        for df in df_list:
            df = compute(df,False)

            file_id = rmn.fstopenall(self.filename, rmn.FST_RW)

            for row in tqdm(df.itertuples(), desc = 'Writing rows') if FSTPY_PROGRESS else df.itertuples():
            # for row in df.itertuples():
                record_path = row.path
                if identical_destination_and_record_path(record_path, self.filename):
                    logging.warning(
                        'StandardFileWriter - record path and output file are identical, adding  new records')
                write_dataframe_record_to_file(file_id, df, row, rewrite)
            rmn.fstcloseall(file_id)


def set_rewrite(df):
    original_df_length = len(df.index)
    dropped_df = df.drop_duplicates(
        subset=['nomvar', 'typvar', 'etiket', 'ip1', 'ip2', 'ip3'], ignore_index=True)
    dropped_df_length = len(dropped_df.index)
    rewrite = True

    if original_df_length != dropped_df_length:
        rewrite = False
        logging.warning('StandardFileWriter - duplicates found, activating rewrite')
    return rewrite


def write_dataframe_record_to_file(file_id, df, row, rewrite):

    data = row.d

    field_dtype = get_field_dtype(row.datyp, row.nbits)

    if str(data.dtype) != field_dtype:
        logging.warning(f'For record at index {row.Index}, nomvar:{row.nomvar} datyp:{row.datyp} nbits:{row.nbits} array.dtype:{row.d.dtype}')  
        logging.warning('Difference in field dtype detected! - check dataframe nbits datyp and array dtype for mismatch')    
        
    rmn.fstecr(file_id, data=np.asfortranarray(data), meta=df.loc[row.Index].to_dict(), rewrite=rewrite)


def identical_destination_and_record_path(record_path, filename):
    return record_path == filename
