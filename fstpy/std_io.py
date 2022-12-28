# -*- coding: utf-8 -*-
import copy
import datetime
import logging
import multiprocessing as mp
import os.path
import pathlib
import time
from ctypes import (POINTER, Structure, c_char_p, c_int, c_int32, c_uint,
                    c_uint32, c_void_p, cast)
# import ctypes as ct
# import numpy.ctypeslib as npc
from typing import Tuple, Type

import numpy as np
import pandas as pd
import rpnpy.librmn.all as rmn
from dask import array as da
from rpnpy.librmn import librmn

from . import _LOCK


def parallel_get_dataframe_from_file(files, get_records_func, n_cores):
    # Step 1: Init multiprocessing.Pool()
    with mp.Pool(processes=n_cores) as pool:
        df_list = pool.starmap(get_records_func, [file for file in files])

    df = pd.concat(df_list, ignore_index=True)
    return df


def get_dataframe_from_file(path: str, query: str = None):
    from .dataframe import add_grid_column
    
    df = get_basic_dataframe(path)

    df = add_grid_column(df)

    hy_df = df.loc[df.nomvar == "HY"]

    df = df.loc[df.nomvar != "HY"]

    if not (query is None):

        query_result_df = df.query(query)

        # get metadata
        df = add_metadata_to_query_results(df, query_result_df, hy_df)

    # check HY count
    df = process_hy(hy_df, df)

    df = add_dask_column(df)

    df = df.drop(['key', 'path', 'shape', 'swa', 'lng'], axis=1, errors='ignore')

    return df

        
               
def open_fst(path: str, mode: str, caller_class: str, error_class: Type):
    file_id = rmn.fstopenall(path, mode)
    logging.info(f'{caller_class} - opening file {path}')
    return file_id


def close_fst(file_id: int, path: str, caller_class: str):
    logging.info(f'{caller_class} - closing file {path}')
    rmn.fstcloseall(file_id)


class GetRecordsFromFile(Exception):
    pass


def add_metadata_to_query_results(df, query_result_df, hy_df) -> pd.DataFrame:
    if df.empty:
        return df
    meta_df = df.loc[df.nomvar.isin(["^>", ">>", "^^", "!!", "!!SF", "P0", "PT", "E1"])]

    query_result_metadata_df = meta_df.loc[meta_df.grid.isin(list(query_result_df.grid.unique()))]

    if (not query_result_df.empty) and (not query_result_metadata_df.empty):
        df = pd.concat([query_result_df, query_result_metadata_df], ignore_index=True)
    elif (not query_result_df.empty) and (query_result_metadata_df.empty):
        df = query_result_df
    elif query_result_df.empty:
        df = query_result_df

    if (not df.empty) and (not hy_df.empty):
        df = pd.concat([df, hy_df], ignore_index=True)

    return df


def process_hy(hy_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Make sure there is only one HY, add it to the dataframe and set its grid

    :param hy_df: dataframe of all hy fields
    :type hy_df: pd.DataFrame
    :param df: original dataframe without hy
    :type df: pd.DataFrame
    :return: modified dataframe with one HY field
    :rtype: pd.DataFrame
    """
    if hy_df.empty or df.empty:
        return df

    # check HY count
    hy_count = hy_df.nomvar.count()

    if hy_count >= 1:
        if hy_count > 1:
            logging.warning(
                'More than one HY in this file! - UNKNOWN BEHAVIOR, continue at your own risk')

        hy_df = pd.DataFrame([hy_df.iloc[0].to_dict()])

        grid = df.grid.unique()[0]
        hy_df['grid'] = grid

        df = pd.concat([df, hy_df], ignore_index=True)
    return df


# written by Micheal Neish creator of fstd2nc
# Lightweight test for FST files.
# Uses the same test for fstd98 random files from wkoffit.c (librmn 16.2).
#
# The 'isFST' test from rpnpy calls c_wkoffit, which has a bug when testing
# many small (non-FST) files.  Under certain conditions the file handles are
# not closed properly, which causes the application to run out of file handles
# after testing ~1020 small non-FST files.

def maybeFST(filename:'str|pathlib.Path') -> bool:
    """Lightweight test to check if file is of FST type (Micheal Neish - fstd2nc)

    :param filename: file to test
    :type filename: str|pathlib.Path
    :return: True if isfile and of FST type, else False
    :rtype: bool
    """
    if not os.path.isfile(filename):
        return False
    with open(filename, 'rb') as f:
        buf = f.read(16)
        if len(buf) < 16:
            return False
        # Same check as c_wkoffit in librmn
        return buf[12:] == b'STDR'


def get_file_modification_time(path):
    file_modification_time = time.ctime(os.path.getmtime(path))
    return datetime.datetime.strptime(file_modification_time, "%a %b %d %H:%M:%S %Y")


def get_lat_lon(df):
    return get_grid_metadata_fields(df, pressure=False, vertical_descriptors=False)


def get_grid_metadata_fields(df,latitude_and_longitude=True, pressure=True, vertical_descriptors=True) -> pd.DataFrame:
    from fstpy.dataframe import add_path_and_key_columns
    new_df = copy.deepcopy(df)
    new_df = add_path_and_key_columns(new_df)
    path_groups = new_df.groupby(new_df.path)
    df_list = []
    #for each files in the df
    for path, rec_df in path_groups:

        if path is None:
            continue

        meta_df = get_all_grid_metadata_fields_from_std_file(path)

        if meta_df.empty:
            # sys.stderr.write('get_grid_metadata_fields - no metatada in file %s\n'%path)
            return pd.DataFrame(dtype=object)
        grid_groups = rec_df.groupby(rec_df.grid)
        #for each grid in the current file
        for _,grid_df in grid_groups:
            this_grid = grid_df.iloc[0]['grid']
            if vertical_descriptors:
                #print('vertical_descriptors')
                vertical_df = meta_df.loc[(meta_df.nomvar.isin(["!!", "HY", "!!SF", "E1"])) & (meta_df.grid==this_grid)]
                df_list.append(vertical_df)
            if pressure:
                #print('pressure')
                pressure_df = meta_df.loc[(meta_df.nomvar.isin(["P0", "PT"])) & (meta_df.grid==this_grid)]
                df_list.append(pressure_df)
            if latitude_and_longitude:
                #print('lati and longi')
                latlon_df = meta_df.loc[(meta_df.nomvar.isin(["^>", ">>", "^^"])) & (meta_df.grid==this_grid)]
                #print(latlon_df)
                df_list.append(latlon_df)
                #print(latlon_df)

    if len(df_list):
        result_df = pd.concat(df_list,ignore_index=True)
        result_df = result_df.drop_duplicates(subset = ['nomvar', 'typvar', 'ni', 'nj', 'nk', 'dateo', 'ip1', 'ip2', 'ip3', 'deet', 'npas', 'grtyp', 'ig1', 'ig2', 'ig3', 'ig4'], ignore_index=True)
        return result_df
    else:
        return pd.DataFrame(dtype=object)


def get_all_grid_metadata_fields_from_std_file(path):
    from .dataframe import add_grid_column
    df = get_basic_dataframe(path)
    df = df.loc[df.nomvar.isin(["^^", ">>", "^>", "!!", "HY", "!!SF", "E1", "P0", "PT"])]
    df = add_grid_column(df)
    df = add_dask_column(df)
    return df

            
class GetMetaDataError(Exception):
    pass


###############################################################################
# Copyright 2017 - Climate Research Division
#                  Environment and Climate Change Canada
#
# This file is part of the "fstd2nc" package.
#
# "fstd2nc" is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# "fstd2nc" is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with "fstd2nc".  If not, see <http://www.gnu.org/licenses/>.
###############################################################################

# """
# Optional helper functions.
# Note: These rely on some assumptions about the internal structures of librmn,
#       and may fail for future libray verisons.
#       These have been tested for librmn 15.2 and 16.2.
# """
# From fnom.h
MAXFILES = 1024


class attributs(Structure):
    _fields_ = [
        ('stream', c_uint, 1), ('std', c_uint, 1), ('burp', c_uint,
                                                    1), ('rnd', c_uint, 1), ('wa', c_uint, 1), ('ftn', c_uint, 1),
        ('unf', c_uint, 1), ('read_only', c_uint, 1), ('old',
                                                       c_uint, 1), ('scratch', c_uint, 1), ('notpaged', c_uint, 1),
        ('pipe', c_uint, 1), ('write_mode', c_uint,
                              1), ('remote', c_uint, 1), ('padding', c_uint, 18),
    ]


class general_file_info(Structure):
    _fields_ = [
        ('file_name', c_char_p),            # complete file name
        ('subname', c_char_p),              # sub file name for cmcarc files
        ('file_type', c_char_p),            # file type and options
        ('iun', c_int),                    # fnom unit number
        ('fd', c_int),                     # file descriptor
        ('file_size', c_int),              # file size in words
        ('eff_file_size', c_int),          # effective file size in words
        ('lrec', c_int),                   # record length when appliable
        ('open_flag', c_int),              # open/close flag
        ('attr', attributs),
    ]


Fnom_General_File_Desc_Table = (
    general_file_info*MAXFILES).in_dll(librmn, 'Fnom_General_File_Desc_Table')

# From rpnmacros.h
word = c_uint32

# From qstdir.h
MAX_XDF_FILES = 1024
ENTRIES_PER_PAGE = 256
MAX_DIR_PAGES = 1024
MAX_PRIMARY_LNG = 16
MAX_SECONDARY_LNG = 8
max_dir_keys = word*MAX_PRIMARY_LNG
max_info_keys = word*MAX_SECONDARY_LNG


class stdf_dir_keys(Structure):
    pass  # defined further below


class xdf_dir_page(Structure):
    _fields_ = [
        ('lng', word, 24), ('idtyp', word,
                            8), ('addr', word, 32),  # XDF record header
        ('reserved1', word, 32), ('reserved2', word, 32),
        ('nxt_addr', word, 32), ('nent', word, 32),
        ('chksum', word, 32), ('reserved3', word, 32),
        ('entry', stdf_dir_keys*ENTRIES_PER_PAGE),
    ]
# idtyp:     id type (usualy 0)
# lng:       header length (in 64 bit units)
# addr:      address of directory page (origin 1, 64 bit units)
# reserved1: idrep (4 ascii char 'DIR0')
# reserved2: reserved (0)
# nxt_addr:  address of next directory page (origin 1, 64 bit units)
# nent:      number of entries in page
# chksum:    checksum (not valid when in core)
# page_no, record_no, file_index: handle templage
# entry:     (real allocated dimension will be ENTRIES_PER_PAGE * primary_len)


class full_dir_page(Structure):
    pass


page_ptr = POINTER(full_dir_page)
full_dir_page._fields_ = [
    ('next_page', page_ptr),
    ('prev_page', page_ptr),
    ('modified', c_int),
    ('true_file_index', c_int),
    ('dir', xdf_dir_page),
]


class file_record(Structure):
    _fields_ = [
        ('lng', word, 24), ('idtyp', word,
                            8), ('addr', word, 32),  # XDF record header
        # primary keys, info keys, data
        ('data', word*2),
    ]


stdf_dir_keys._fields_ = [
    ('lng', word, 24), ('select', word, 7), ('deleted', word, 1), ('addr', word, 32),
    ('nbits', word, 8), ('deet', word, 24), ('gtyp', word, 8), ('ni', word, 24),
    ('datyp', word, 8), ('nj', word, 24), ('ubc', word, 12), ('nk', word, 20),
    ('pad7', word, 6), ('npas', word, 26), ('ig2a', word, 8), ('ig4', word, 24),
    ('ig2b', word, 8), ('ig1', word, 24), ('ig2c', word, 8), ('ig3', word, 24),
    ('pad1', word, 2), ('etik15', word,
                        30), ('pad2', word, 2), ('etik6a', word, 30),
    ('pad3', word, 8), ('typvar', word, 12), ('etikbc',
                                              word, 12), ('pad4', word, 8), ('nomvar', word, 24),
    ('levtyp', word, 4), ('ip1', word, 28), ('pad5', word, 4), ('ip2', word, 28),
    ('pad6', word, 4), ('ip3', word, 28), ('date_stamp', word, 32),
]


class key_descriptor(Structure):
    _fields_ = [
        ('ncle', word, 32), ('reserved', word, 8), ('tcle',
                                                    word, 6), ('lcle', word, 5), ('bit1', word, 13),
    ]


class file_header(Structure):
    _fields_ = [
        ('lng', word, 24), ('idtyp', word, 8), ('addr',
                                                word, 32),  # standard XDF record header
        ('vrsn', word),     ('sign', word),  # char[4]
        ('fsiz', word, 32), ('nrwr', word, 32),
        ('nxtn', word, 32), ('nbd', word, 32),
        ('plst', word, 32), ('nbig', word, 32),
        ('lprm', word, 16), ('nprm', word,
                             16), ('laux', word, 16), ('naux', word, 16),
        ('neff', word, 32), ('nrec', word, 32),
        ('rwflg', word, 32), ('reserved', word, 32),
        ('keys', key_descriptor*1024),
    ]
# idtyp:     id type (usualy 0)
# lng:       header length (in 64 bit units)
# addr:      address (exception: 0 for a file header)
# vrsn:      XDF version
# sign:      application signature
# fsiz:      file size (in 64 bit units)
# nrwr:      number of rewrites
# nxtn:      number of extensions
# nbd:       number of directory pages
# plst:      address of last directory page (origin 1, 64 bit units)
# nbig:      size of biggest record
# nprm:      number of primary keys
# lprm:      length of primary keys (in 64 bit units)
# naux:      number of auxiliary keys
# laux:      length of auxiliary keys
# neff:      number of erasures
# nrec:      number of valid records
# rwflg:     read/write flag
# reserved:  reserved
# keys:      key descriptor table


class file_table_entry(Structure):
    _fields_ = [
        ('dir_page', page_ptr*MAX_DIR_PAGES),  # pointer to directory pages
        # pointer to current directory page
        ('cur_dir_page', page_ptr),
        # pointer to primary key building function
        ('build_primary', c_void_p),
        # pointer to info building function
        ('build_info', c_void_p),
        ('scan_file', c_void_p),            # pointer to file scan function
        # pointer to record filter function
        ('file_filter', c_void_p),
        # pointer to current directory entry
        ('cur_entry', POINTER(word)),
        ('header', POINTER(file_header)),          # pointer to file header
        ('nxtadr', c_int32),                # next write address (in word units)
        ('primary_len', c_int),
        # length in 64 bit units of primary keys (including 64 bit header)
        ('info_len', c_int),  # length in 64 bit units of info keys
        # file index to next linked file,-1 if none
        ('link', c_int),
        ('cur_info', POINTER(general_file_info)),
        # pointer to current general file desc entry
        # FORTRAN unit number, -1 if not open, 0 if C file
        ('iun', c_int),
        # index into file table, -1 if not open
        ('file_index', c_int),
        ('modified', c_int),                 # modified flag
        # number of allocated directory pages
        ('npages', c_int),
        ('nrecords', c_int),                 # number of records in file
        ('cur_pageno', c_int),               # current page number
        # record number within current page
        ('page_record', c_int),
        # number of records in current page
        ('page_nrecords', c_int),
        ('file_version', c_int),             # version number
        ('valid_target', c_int),             # last search target valid flag
        ('xdf_seq', c_int),                  # file is sequential xdf
        # last position valid flag (seq only)
        ('valid_pos', c_int),
        # current address (WA, sequential xdf)
        ('cur_addr', c_int),
        # address (WA) of first record (seq xdf)
        ('seq_bof', c_int),
        ('fstd_vintage_89', c_int),          # old standard file flag
        # header & primary keys for last record
        ('head_keys', max_dir_keys),
        # info for last read/written record
        ('info_keys', max_info_keys),
        ('cur_keys', max_dir_keys),        # keys for current operation
        ('target', max_dir_keys),          # current search target
        # permanent search mask for this file
        ('srch_mask', max_dir_keys),
        ('cur_mask', max_dir_keys),        # current search mask for this file
    ]


file_table_entry_ptr = POINTER(file_table_entry)

librmn.file_index.argtypes = (c_int,)
librmn.file_index.restype = c_int
file_table = (file_table_entry_ptr*MAX_XDF_FILES).in_dll(librmn, 'file_table')


def get_data(path, key, cache={}):
    with _LOCK:
        # Check if file needs to be opened.
        if path not in cache:
            # Allow for a small number of files to remain open for speedier access.
            if len(cache) > 10:
                for _, unit in cache.items():
                    rmn.fstcloseall(unit)
                cache.clear()
            cache[path] = rmn.fstopenall(path)
        iun = cache[path]
        key = ((key >> 10) << 10) + librmn.file_index(iun)
        return rmn.fstluk(key)['d']

######################################################################
#
# Direct reading of FST data without fstluk (which is not thread safe).
# Allows for better threading performance by avoiding the need to lock.
#

# librmn.compact_float.argtypes = (npc.ndpointer(dtype='int32'), npc.ndpointer(dtype='int32'), npc.ndpointer(dtype='int32'), ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.POINTER(ct.c_double))
# librmn.compact_double.argtypes = (npc.ndpointer(dtype='int32'), npc.ndpointer(dtype='int32'), npc.ndpointer(dtype='int32'), ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.POINTER(ct.c_double))
# librmn.compact_integer.argtypes = (npc.ndpointer(dtype='int32'), ct.c_void_p, npc.ndpointer(dtype='int32'), ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int)
# librmn.ieeepak_.argtypes = (npc.ndpointer(dtype='int32'), npc.ndpointer(dtype='int32'), ct.POINTER(ct.c_int), ct.POINTER(ct.c_int), ct.POINTER(ct.c_int), ct.POINTER(ct.c_int), ct.POINTER(ct.c_int))
# librmn.compact_char.argtypes = (npc.ndpointer(dtype='int32'), ct.c_void_p, npc.ndpointer(dtype='int32'), ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int)
# librmn.c_armn_uncompress32.argtypes = (npc.ndpointer(dtype='int32'), npc.ndpointer(dtype='int32'), ct.c_int, ct.c_int, ct.c_int, ct.c_int)
# librmn.armn_compress.argtypes = (npc.ndpointer(dtype='int32'),ct.c_int,ct.c_int,ct.c_int,ct.c_int,ct.c_int)
# librmn.c_float_unpacker.argtypes = (npc.ndpointer(dtype='int32'),npc.ndpointer(dtype='int32'),npc.ndpointer(dtype='int32'),ct.c_int,ct.POINTER(ct.c_int))

# def get_data_fast(path, swa, lng):
#     import rpnpy.librmn.all as rmn
#     import numpy as np
#     with open(path,'rb') as f:
#         f.seek(swa*8-8,0)
#         data = np.fromfile(f,'B',lng*4)
#     data = data.view('>i4').astype('i4')
#     ni, nj, nk = data[3]>>8, data[4]>>8, data[5]>>12
#     nelm = ni*nj*nk
#     datyp = int(data[4]%256) & 191  # Ignore +64 mask.
#     nbits = int(data[2]%256)
#     dtype = rmn.dtype_fst2numpy (datyp, nbits)
#     if nbits <= 32:
#         work = np.empty(nelm,'int32')
#     else:
#         work = np.empty(nelm,'int64').view('int32')
#     # Strip header
#     data = data[20:]
#     # Extend data buffer for in-place decompression.
#     if datyp in (129,130,134):
#         d = np.empty(nelm + 100, dtype='int32')
#         d[:len(data)] = data
#         data = d
#     shape = (nj,ni)
#     ni = ct.c_int(ni)
#     nj = ct.c_int(nj)
#     nk = ct.c_int(nk)
#     nelm = ct.c_int(nelm)
#     npak = ct.c_int(-nbits)
#     nbits = ct.c_int(nbits)
#     zero = ct.c_int(0)
#     one = ct.c_int(1)
#     two = ct.c_int(2)
#     tempfloat = ct.c_double(99999.0)

#     if datyp == 0:
#         work = data
#     elif datyp == 1:
#         if nbits.value <= 32:
#             librmn.compact_float(work, data, data[3:], nelm, nbits, 24, 1, 2, 0, ct.byref(tempfloat))
#         else:
#             raise Exception
#             librmn.compact_double(work, data, data[3:], nelm, nbits, 24, 1, 2, 0, ct.byref(tempfloat))
#     elif datyp == 2:
#         librmn.compact_integer(work, None, data, nelm, nbits, 0, 1, 2)
#     elif datyp == 3:
#         raise Exception
#     elif datyp == 4:
#         librmn.compact_integer(work, None, data, nelm, nbits, 0, 1, 4)
#     elif datyp == 5:
#         librmn.ieeepak_(work, data, ct.byref(nelm), ct.byref(one), ct.byref(npak), ct.byref(zero), ct.byref(two))
#     elif datyp == 6:
#         librmn.c_float_unpacker(work, data, data[3:], nelm, ct.byref(nbits));
#     elif datyp == 7:
#         ier = librmn.compact_char(work, None, data, nelm, 8, 0, 1, 10)
#         work = work.view('B')[:len(work)] #& 127
#     elif datyp == 8:
#         raise Exception
#     elif datyp == 129:
#         librmn.armn_compress(data[5:],ni,nj,nk,nbits,2)
#         librmn.compact_float(work,data[1:],data[5:],nelm,nbits.value+64*max(16,nbits.value),0,1,2,0,ct.byref(tempfloat))
#     elif datyp == 130:
#         librmn.armn_compress(data[1:],ni,nj,nk,nbits,2)
#         work[:] = data[1:].astype('>i4').view('>H')[:nelm.value]
#     elif datyp == 133:
#         librmn.c_armn_uncompress32(work, data[1:], ni, nj, nk, nbits)
#     elif datyp == 134:
#         librmn.armn_compress(data[4:],ni,nj,nk,nbits,2);
#         librmn.c_float_unpacker(work,data[1:],data[4:],nelm,ct.byref(nbits))
#     else:
#         raise Exception(datyp)
#     return work.view(dtype)[:nelm.value].reshape(shape).T
# #
# ######################################################################


# def add_dask_column2(df:pd.DataFrame) -> pd.DataFrame:
#     """Adds the 'd' column as dask arrays to a basic dataframe of meta data only, path and key columns have to be present in the DataFrame

#     :param df: input dataframe
#     :type df: pd.DataFrame
#     :return: modified Dataframe with added 'd' column
#     :rtype: pd.DataFrame
#     """
#     arrays = []
#     for row in df.itertuples():
#         path = row.path
#         key = row.key
#         swa = row.swa
#         lng = row.lng
#         # Unique identifier for this record.
#         name = ''.join([path,":",str(key)])
#         # How to read the data for this record.
#         shape = row.shape
#         dsk = {(name, 0, 0): (get_data_fast, path, swa, lng)}
#         field_dtype = get_field_dtype(row.datyp, row.nbits)
#         # Size of the record.
#         chunks = [(s,) for s in shape]
#         arrays.append(da.Array(dsk, name, chunks, field_dtype))
#     # Very *carefully* add to pandas.
#     # Need to avoid triggering evaluation of the dask arrays via numpy.
#     d = np.zeros(len(arrays), dtype=object)
#     for i in range(len(d)):
#         d[i] = arrays[i]
#     df['d'] = d
#     return df

def add_dask_column(df:pd.DataFrame) -> pd.DataFrame:
    """Adds the 'd' column as dask arrays to a basic dataframe of meta data only, path and key columns have to be present in the DataFrame

    :param df: input dataframe
    :type df: pd.DataFrame
    :return: modified Dataframe with added 'd' column
    :rtype: pd.DataFrame
    """
    if df.empty:
        return df
    arrays = []
    for row in df.itertuples():
        path = row.path
        key = row.key
        # swa = row.swa
        # lng = row.lng
        # Unique identifier for this record.
        name = ''.join([path,":",str(key)])
        # How to read the data for this record.
        shape = row.shape
        dsk = {(name, 0, 0): (get_data, path, key)}
        field_dtype = get_field_dtype(row.datyp, row.nbits)
        # Size of the record.
        chunks = [(s,) for s in shape]
        arrays.append(da.Array(dsk, name, chunks, field_dtype))
    # Very *carefully* add to pandas.
    # Need to avoid triggering evaluation of the dask arrays via numpy.
    d = np.zeros(len(arrays), dtype=object)
    for i in range(len(d)):
        d[i] = arrays[i]
    df['d'] = d
    return df


def get_field_dtype(datyp, nbits):
    field_dtype = 'float32'
    if (datyp in [1, 5, 6, 133, 134]) and (nbits <= 32):
        field_dtype = 'float32'
    elif (datyp in [1, 5, 6, 133, 134]) and (nbits > 32):
        field_dtype = 'float64'
    elif (datyp in [2, 130]):
        if nbits > 1:
            field_dtype = 'int32'
        elif nbits == 1:
            field_dtype = 'uint32'
    return field_dtype


class GetBasicDataFrameError(Exception):
    pass

# Extract parameters for *all* records.
# Returns a dictionary similar to fstprm, only the entries are
# vectorized over all records instead of 1 record at a time.
# NOTE: This includes deleted records as well.  You can filter them out using
# the 'dltf' flag.

def get_basic_dataframe(path:str) -> pd.DataFrame:
    """Creates a dataframe of all non deleted records in an FST file, does not include data 'd'

    :param path: path of file to load
    :type path: str
    :return: dataframe of all non deleted records in an FST file
    :rtype: pd.DataFrame
    """
    # file_id, f_mod_time = rmn.fstopenall(path)
    file_id = open_fst(path, rmn.FST_RO, 'get_basic_dataframe', 'GetBasicDataFrameError')

    # Get the raw (packed) parameters.
    index = librmn.file_index(file_id)
    raw = []
    file_index_list = []
    pageno_list = []
    recno_list = []
    while index >= 0:
        f = file_table[index].contents
        for pageno in range(f.npages):
            page = f.dir_page[pageno].contents
            params = cast(page.dir.entry, POINTER(c_uint32))
            params = np.ctypeslib.as_array(
                params, shape=(ENTRIES_PER_PAGE, 9, 2))
            nent = page.dir.nent
            raw.append(params[:nent])
            recno_list.extend(list(range(nent)))
            pageno_list.extend([pageno]*nent)
            file_index_list.extend([index]*nent)
        index = f.link
    raw = np.concatenate(raw)
    # Start unpacking the pieces.
    # Reference structure (from qstdir.h):
    # 0      word deleted:1, select:7, lng:24, addr:32;
    # 1      word deet:24, nbits: 8, ni:   24, gtyp:  8;
    # 2      word nj:24,  datyp: 8, nk:   20, ubc:  12;
    # 3      word npas: 26, pad7: 6, ig4: 24, ig2a:  8;
    # 4      word ig1:  24, ig2b:  8, ig3:  24, ig2c:  8;
    # 5      word etik15:30, pad1:2, etik6a:30, pad2:2;
    # 6      word etikbc:12, typvar:12, pad3:8, nomvar:24, pad4:8;
    # 7      word ip1:28, levtyp:4, ip2:28, pad5:4;
    # 8      word ip3:28, pad6:4, date_stamp:32;
    nrecs = raw.shape[0]

    out = {}
    out['nomvar'] = np.empty(nrecs, dtype='|S4')
    out['typvar'] = np.empty(nrecs, dtype='|S2')
    out['etiket'] = np.empty(nrecs, dtype='|S12')
    out['ni'] = np.empty(nrecs, dtype='int32')
    out['nj'] = np.empty(nrecs, dtype='int32')
    out['nk'] = np.empty(nrecs, dtype='int32')
    out['dateo'] = np.empty(nrecs, dtype='int32')
    out['ip1'] = np.empty(nrecs, dtype='int32')
    out['ip2'] = np.empty(nrecs, dtype='int32')
    out['ip3'] = np.empty(nrecs, dtype='int32')
    out['deet'] = np.empty(nrecs, dtype='int32')
    out['npas'] = np.empty(nrecs, dtype='int32')
    out['datyp'] = np.empty(nrecs, dtype='ubyte')
    out['nbits'] = np.empty(nrecs, dtype='byte')
    out['grtyp'] = np.empty(nrecs, dtype='|S1')
    out['ig1'] = np.empty(nrecs, dtype='int32')
    out['ig2'] = np.empty(nrecs, dtype='int32')
    out['ig3'] = np.empty(nrecs, dtype='int32')
    out['ig4'] = np.empty(nrecs, dtype='int32')
    out['datev'] = np.empty(nrecs, dtype='int32')

    out['lng'] = np.empty(nrecs, dtype='int32')
    out['dltf'] = np.empty(nrecs, dtype='ubyte')
    out['swa'] =  np.empty(nrecs, dtype='uint32')
    out['ubc'] = np.empty(nrecs, dtype='uint16')
    # out['xtra1'] = np.empty(nrecs, dtype='uint32')
    # out['xtra2'] = np.empty(nrecs, dtype='uint32')
    # out['xtra3'] = np.empty(nrecs, dtype='uint32')
    out['key'] = np.empty(nrecs, dtype='int32')

    temp8 = np.empty(nrecs, dtype='ubyte')
    temp32 = np.empty(nrecs, dtype='int32')

    np.divmod(raw[:, 0, 0], 2**24, temp8, out['lng'])
    out['lng'] *= 2  # Convert from 64-bit word lengths to 32-bit.
    np.divmod(temp8, 128, out['dltf'], temp8)
    out['swa'][:] = raw[:,0,1]
    np.divmod(raw[:, 1, 0], 256, out['deet'], out['nbits'])
    np.divmod(raw[:, 1, 1], 256, out['ni'], out['grtyp'].view('ubyte'))
    np.divmod(raw[:, 2, 0], 256, out['nj'], out['datyp'])
    np.divmod(raw[:, 2, 1], 4096, out['nk'], out['ubc'])
    out['npas'][:] = raw[:, 3, 0]//64
    np.divmod(raw[:, 3, 1], 256, out['ig4'], temp32)
    out['ig2'][:] = (temp32 << 16)  # ig2a
    np.divmod(raw[:, 4, 0], 256, out['ig1'], temp32)
    out['ig2'] |= (temp32 << 8)  # ig2b
    np.divmod(raw[:, 4, 1], 256, out['ig3'], temp32)
    out['ig2'] |= temp32  # ig2c
    etik15 = raw[:, 5, 0]//4
    etik6a = raw[:, 5, 1]//4
    et = raw[:, 6, 0]//256
    etikbc, _typvar = divmod(et, 4096)
    _nomvar = raw[:, 6, 1]//256
    np.divmod(raw[:, 7, 0], 16, out['ip1'], temp8)
    out['ip2'][:] = raw[:, 7, 1]//16
    out['ip3'][:] = raw[:, 8, 0]//16
    date_stamp = raw[:, 8, 1]
    # Reassemble and decode.
    # (Based on fstd98.c)
    etiket_bytes = np.empty((nrecs, 12), dtype='ubyte')

    for i in range(5):
        etiket_bytes[:, i] = ((etik15 >> ((4-i)*6)) & 0x3f) + 32

    for i in range(5, 10):
        etiket_bytes[:, i] = ((etik6a >> ((9-i)*6)) & 0x3f) + 32

    etiket_bytes[:, 10] = ((etikbc >> 6) & 0x3f) + 32
    etiket_bytes[:, 11] = (etikbc & 0x3f) + 32
    out['etiket'][:] = etiket_bytes.flatten().view('|S12')
    nomvar_bytes = np.empty((nrecs, 4), dtype='ubyte')

    for i in range(4):
        nomvar_bytes[:, i] = ((_nomvar >> ((3-i)*6)) & 0x3f) + 32

    out['nomvar'][:] = nomvar_bytes.flatten().view('|S4')
    typvar_bytes = np.empty((nrecs, 2), dtype='ubyte')
    typvar_bytes[:, 0] = ((_typvar >> 6) & 0x3f) + 32
    typvar_bytes[:, 1] = ((_typvar & 0x3f)) + 32
    out['typvar'][:] = typvar_bytes.flatten().view('|S2')
    out['datev'][:] = (date_stamp >> 3) * 10 + (date_stamp & 0x7)
    # Note: this dateo calculation is based on my assumption that
    # the raw stamps increase in 5-second intervals.
    # Doing it this way to avoid a gazillion calls to incdat.
    date_stamp = date_stamp - (out['deet']*out['npas'])//5
    out['dateo'][:] = (date_stamp >> 3) * 10 + (date_stamp & 0x7)

    #   out['xtra1'][:] = out['datev']
    #   out['xtra2'][:] = 0
    #   out['xtra3'][:] = 0
    # Calculate the handles (keys)
    # Based on "MAKE_RND_HANDLE" macro in qstdir.h.
    out['nomvar'] = np.char.strip(out['nomvar'].astype('str'))
    out['typvar'] = np.char.strip(out['typvar'].astype('str'))
    out['etiket'] = np.char.strip(out['etiket'].astype('str'))
    out['grtyp'] = np.char.strip(out['grtyp'].astype('str'))

    out['key'][:] = (np.array(file_index_list) & 0x3FF) | (
        (np.array(recno_list) & 0x1FF) << 10) | ((np.array(pageno_list) & 0xFFF) << 19)

    close_fst(file_id, path, 'get_basic_dataframe')

    df = pd.DataFrame(out)

    df['path'] = path

    df = df.loc[df.dltf == 0]
    df = df.drop(labels=['dltf', 'ubc'], axis=1)

    df['shape'] = pd.Series(zip(df.ni.to_numpy(),df.nj.to_numpy()),dtype='object').to_numpy()

    return df

class DecodeIpError(Exception):
    pass

def kind_to_string(kind:int) -> str:
    return '' if kind in [-1, 3, 15, 17, 100] else rmn.kindToString(kind).strip()

def decode_ip123(nomvar:str, ip1: int, ip2: int, ip3: int) -> 'Tuple(dict, dict, dict)':
    ip_info = {'v1': 0., 'kind': -1, 'kinds': ''}
    
    if nomvar in ['>>', '^^', '^>', '!!']:
        ip1_info = copy.deepcopy(ip_info)
        ip1_info['v1'] = float(ip1)
        ip1_info['kind'] = 100

        ip2_info = copy.deepcopy(ip_info)
        ip2_info['v1'] = float(ip2)
        ip2_info['kind'] = 100

        ip3_info = copy.deepcopy(ip_info)
        ip3_info['v1'] = float(ip3)
        ip3_info['kind'] = 100
        
    else:
        ip1_info = copy.deepcopy(ip_info)
        ip1_info['v1'], ip1_info['kind'] = rmn.convertIp(rmn.CONVIP_DECODE, ip1)
        ip1_info['kinds'] = kind_to_string(ip1_info['kind'])

        ip2_info = copy.deepcopy(ip_info)
        ip2_info['v1'], ip2_info['kind'] = rmn.convertIp(rmn.CONVIP_DECODE, ip2)
        ip2_info['kinds'] = kind_to_string(ip2_info['kind'])
        if (ip2 >= 32768): # Verifie si IP2 est encode
            if (ip2_info['kind'] != 10):
                raise DecodeIpError(f'Invalid kind value for ip2 {ip2_info["kind"]} != 10')
        else:
            ip2_info['kind'] = 10
            ip2_info['kinds'] = kind_to_string(ip2_info['kind'])

        ip3_info = copy.deepcopy(ip_info)
        ip3_info['v1'], ip3_info['kind'] = rmn.convertIp(rmn.CONVIP_DECODE, ip3)
        ip3_info['kinds'] = kind_to_string(ip3_info['kind'])
        if (ip3 < 32768): # Verifie si IP3 est encode
            ip3_info['kind'] = 100
            ip3_info['kinds'] = kind_to_string(ip3_info['kind'])

        if nomvar not in ['>>', '^^', '^>', '!!', 'HY', 'P0', 'PT']:
            # Nous n'avons pas de champs speciaux
            if (ip3 >= 32768):
                if (ip3_info['kind'] == ip2_info['kind']): # On a un intervalle de temps
                    v1 = ip3_info['v1']
                    v2 = ip2_info['v1']
                    ip2_info['v1'] = v1
                    ip2_info['v2'] = v2
                elif (ip3_info['kind'] == ip1_info['kind']): # On a un intervalle sur les hauteurs
                    v1 = ip1_info['v1']
                    v2 = ip3_info['v1']
                    ip1_info['v1'] = v1
                    ip1_info['v2'] = v2

    return ip1_info, ip2_info, ip3_info
