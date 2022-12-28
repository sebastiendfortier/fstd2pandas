# -*- coding: utf-8 -*-
import pandas as pd
import os
import numpy as np
import datetime
from typing import Final
from .utils import ArrayIs3dError, CsvArray
from .std_enc import create_encoded_dateo, create_encoded_ip1
from .dataframe import add_grid_column
import rpnpy.librmn.all as rmn


BASE_COLUMNS = ['nomvar', 'typvar', 'etiket', 'level', 'dateo', 'ip1', 'ip2', 'ip3',
                'deet', 'npas', 'datyp', 'nbits', 'grtyp', 'ig1', 'ig2', 'ig3', 'ig4', 'd','datev','ni','nj','nk']


IP1_KIND: Final[int] = 3
"""ip1 kind used in levels"""

NOMVAR_MIN_LEN: Final[int] = 2
"""minimum length for nomvar values"""

NOMVAR_MAX_LEN: Final[int] = 4
"""maximum length for nomvar values"""

TYPVAR_MIN_LEN: Final[int] = 1
"""minimum length for typvar values"""

TYPVAR_MAX_LEN: Final[int] = 2
"""maximum length for typvar values"""

ETIKET_MIN_LEN: Final[int] = 1
"""minimum length for etiket values"""

ETIKET_MAX_LEN: Final[int] = 12
"""maximum length for etiket values"""

NBITS_DEFAULT: Final[int] = 24
DATYP_DEFAULT: Final[int] = 1
GRTYP_DEFAULT: Final[str] = "X"
TYPVAR_DEFAULT: Final[str] = "X"
IP2_DEFAULT: Final[int] = 0
IP3_DEFAULT: Final[int] = 0
IG1_DEFAULT: Final[int] = 0
IG2_DEFAULT: Final[int] = 0
IG3_DEFAULT: Final[int] = 0
IG4_DEFAULT: Final[int] = 0
ETIKET_DEFAULT: Final[str] = "CSVREADER"
DEET_DEFAULT: Final[int] = 0
NPAS_DEFAULT: Final[int] = 0


class CsvFileReaderError(Exception):
    pass


class MinimalColumnsError(Exception):
    pass


class ColumnsNotValidError(Exception):
    pass


class Ip1andLevelExistsError(Exception):
    pass


class DimensionError(Exception):
    pass


class NomVarLengthError(Exception):
    pass


class TypVarLengthError(Exception):
    pass


class EtiketVarLengthError(Exception):
    pass


class CsvFileReader:
    """Read a csv file and convert it to a pandas dataframe.

    :param path: path of the csv file to read
    :type path: str
    

    Algorithm:

    Read a file that must have the following form:
    
    +---------+------------+-----------+-------------------------------------+
    | nomvar  | etiket     |   level   |              d                      |
    +=========+============+===========+=====================================+
    | CSV     | CSVREADER  |    1.0    | 11.1,22.2;33.3,44.4;55.5,66.6       |
    +---------+------------+-----------+-----------+-------------------------+
    | CSV     | CSVREADER  |    0.0    | 77.7,88.8;99.9,100.10;110.11,120.12 |
    +---------+------------+-----------+-------------------------------------+


    - The d column is composed of floats and the ";" means one of the line of the level is done
    - You can't provide an ip1 column and a level column at the same time in your dataframe
    - One line of a single level represent the x axis (row)
    - The values inside a single line are the y axis (column)
    - ni and nj will be derived from the provided array shape in the d column
    - nk is always equal to 1
    - arrays in rows sharing the same metadata columns (same nomvar, etiket, etc.), must have the same dimensions
    - comments are permitted on separate lines than the csv values 


    - Admissible columns: 

        +---------+--------+------------------------------------------------------+
        | column  | type   |                   details                            | 
        +=========+========+======================================================+
        | nomvar  | str    |                  variable name                       |
        +---------+--------+------------------------------------------------------+
        | typvar  | str    |                 type of field                        | 
        +---------+--------+------------------------------------------------------+
        | etiket  | str    |                   label                              | 
        +---------+--------+------------------------------------------------------+
        | level   | str    |               value that helps get ip1               | 
        +---------+--------+------------------------------------------------------+
        | ip1     | int32  |                 vertical level                       | 
        +---------+--------+------------------------------------------------------+
        | ip2     | int32  |                 forecast hour                        | 
        +---------+--------+------------------------------------------------------+
        | ip3     | int32  |                user defined identifier               | 
        +---------+--------+------------------------------------------------------+
        | datyp   | int32  |                    data type                         | 
        +---------+--------+------------------------------------------------------+
        | nbits   | int32  |    number of bits kept for the elements of the field | 
        +---------+--------+------------------------------------------------------+
        | grtyp   | str    |   type of geographical projection                    | 
        +---------+--------+------------------------------------------------------+
        | d       | str    |    data column                                       | 
        +---------+--------+------------------------------------------------------+



    - If not already provided these columns will be added:

        +---------+--------+------------------------------------------------------+
        | column  | type   |                   details                            | 
        +=========+========+======================================================+
        | nomvar  | str    |                  variable name                       |
        +---------+--------+------------------------------------------------------+
        | typvar  | str    |                 type of field                        | 
        +---------+--------+------------------------------------------------------+
        | etiket  | str    |                   label                              | 
        +---------+--------+------------------------------------------------------+
        | dateo   | int32  |               date of observation                    | 
        +---------+--------+------------------------------------------------------+
        | ip1     | int32  |                 vertical level                       | 
        +---------+--------+------------------------------------------------------+
        | ip2     | int32  |                 forecast hour                        | 
        +---------+--------+------------------------------------------------------+
        | ip3     | int32  |                user defined identifier               | 
        +---------+--------+------------------------------------------------------+
        | deet    | int32  | Length of a time step in seconds datev constant      |  
        +---------+--------+------------------------------------------------------+
        | npas    | int32  | time step number datev constant unless keep_dateo    | 
        +---------+--------+------------------------------------------------------+
        | datyp   | int32  |                    data type                         |
        +---------+--------+------------------------------------------------------+
        | nbits   | int32  |    number of bits kept for the elements of the field | 
        +---------+--------+------------------------------------------------------+
        | grtyp   | str    |   type of geographical projection                    | 
        +---------+--------+------------------------------------------------------+
        | ni      | int32  |    first dimension of the data field                 | 
        +---------+--------+------------------------------------------------------+
        | nj      | int32  |    second dimension of the data field                | 
        +---------+--------+------------------------------------------------------+
        | nk      | int32  |    third dimension of the data field                 | 
        +---------+--------+------------------------------------------------------+
        | ig1     | int32  |                first grid descriptor                 | 
        +---------+--------+------------------------------------------------------+
        | ig2     | int32  |               second grid descriptor                 | 
        +---------+--------+------------------------------------------------------+
        | ig3     | int32  |               third grid descriptor                  | 
        +---------+--------+------------------------------------------------------+
        | ig4     | int32  |                fourth grid descriptor                | 
        +---------+--------+------------------------------------------------------+
        | datev   | int32  |                date of validation                    | 
        +---------+--------+------------------------------------------------------+
        | grid    | str    |                                                      | 
        +---------+--------+------------------------------------------------------+
    """

    def __init__(self, path, encode_ip1=True):
        self.path = path
        self.encode_ip1 = encode_ip1
        if not os.path.exists(self.path):
            raise CsvFileReaderError('Path does not exist\n')

    def to_pandas(self) -> pd.DataFrame:
        """Read the csv file , verify the existence of headers that are valid and add the missing columns in the dataframe.

        :return: dataframe completed
        :rtype: pd.DataFrame
        """
        self.df = pd.read_csv(self.path, comment="#")
        self.df.columns = self.df.columns.str.replace(' ', '')
        if(self.verify_headers()):
            self.add_missing_columns()
            self.check_columns()
            self.df = add_grid_column(self.df)
            return self.df

    def count_char(self, s):
        """Count the number of characters in a string. This function check the length of a string in a specific column of the dataframe

        :param s: the name of the column
        :type s: str
        :return: a list with the counts of the number of characters inside a string in the differents rows of a single column
        """
        array_list = []
        for i in self.df.index:
            a = len(self.df.at[i, s])
            array_list.append(a)
        return array_list

    def check_nomvar_char_length(self):
        """Check that the length of the column nomvar is always between 2 and 4 characters for the whole dataframe 

        :raises NomVarLengthError: the nomvar values does not have the correct length
        """

        a = self.count_char(s="nomvar")
        for i in a:
            if (i < NOMVAR_MIN_LEN or i > NOMVAR_MAX_LEN):
                raise NomVarLengthError(f"the variable nomvar should have between {NOMVAR_MIN_LEN} and {NOMVAR_MAX_LEN} characters")

    def check_typvar_char_length(self):
        """Check that the length of the column typvar is always between 1 and 2 characters for the whole dataframe

        :raises TypVarLengthError: the typvar values does not have the correct length
        """
        a = self.count_char(s="typvar")
        for i in a:
            if (i < TYPVAR_MIN_LEN or i > TYPVAR_MAX_LEN):
                raise TypVarLengthError(f"the variable typvar should have between {TYPVAR_MIN_LEN} and {TYPVAR_MAX_LEN} characters")

    def check_etiket_char_length(self):
        """Check that the length of the column etiket is always between 1 and 12 characters for the whole dataframe 

        :raises EtiketVarLengthError: the etiket values does not have the correct length
        """
        a = self.count_char(s="etiket")
        for i in a:
            if (i < ETIKET_MIN_LEN or i > ETIKET_MAX_LEN):
                raise EtiketVarLengthError(f"the variable etiket should have between {ETIKET_MIN_LEN} and {ETIKET_MAX_LEN} characters")

    def verify_headers(self):
        """Verify the file header

        :return: self.has_minimal_columns() and self.valid_columns()
        :rtype: Boolean
        """
        return self.has_minimal_columns() and self.valid_columns()

    def add_missing_columns(self):
        """Add the missings columns to the dataframe 
        """
        self.add_nbits()
        self.add_datyp()
        self.add_grtyp()
        self.add_typvar()
        self.add_ip2_ip3()
        self.add_ig()
        self.add_etiket()
        self.add_ip1()
        self.add_array_dimensions()
        self.add_deet()
        self.add_npas()
        self.add_date()
        self.to_numpy_array()

    def check_columns(self):
        """Check the types of the columns, the dimensions of the differents arrays and the length of the values of nomvar,
        etiket and typvar in the dataframe"""

        self.change_column_dtypes()
        self.check_array_dimensions()
        self.check_nomvar_char_length()
        self.check_typvar_char_length()
        self.check_etiket_char_length()

    def has_minimal_columns(self):
        """Verify that I have the minimum amount of headers 

        :raises MinimalColumnsError: the necessary headers are not present in the dataframe
        :return: True
        :rtype: bool
        """

        list_of_hdr_names = self.df.columns.tolist()

        if set(['nomvar', 'd', 'level']).issubset(list_of_hdr_names) or set(['nomvar', 'd', 'ip1']).issubset(list_of_hdr_names):
            return True
        else:
            raise MinimalColumnsError('Your csv file does not have the necessary columns to proceed! Check that you have at least nomvar,d and level or ip1 as columns in your csv file')

    def valid_columns(self):
        """Check that all the provided columns are valid and are present in BASE_COLUMN list

        :raises ColumnsNotValidError: the column names are not valid
        :return: True
        :rtype: Boolean
        """
        all_the_cols = BASE_COLUMNS
        all_the_cols.sort()
        list_of_hdr_names = self.df.columns.tolist()
        list_of_hdr_names.sort()

        set1 = set(list_of_hdr_names)
        set2 = set(BASE_COLUMNS)

        if(len(list_of_hdr_names) < len(BASE_COLUMNS) or len(list_of_hdr_names) > len(BASE_COLUMNS) ):
            is_subset = set1.issubset(set2)
            if(is_subset):
                return True
            else:
                raise ColumnsNotValidError(f'The headers in the csv file are not valid. Make sure that the columns names are present in {BASE_COLUMNS}. The current columns are {list_of_hdr_names}')

        if(len(list_of_hdr_names) == len(BASE_COLUMNS)):
            if all_the_cols == list_of_hdr_names:
                return True
            else:
                raise ColumnsNotValidError(f'The headers in the csv file are not valid. Make sure that the columns names are present in {BASE_COLUMNS}. The current columns are {list_of_hdr_names}')
        else:
            raise ColumnsNotValidError(f'The headers in the csv file are not valid you have too many columns. The current columns are {list_of_hdr_names}')

    def column_exists(self, col):
        """Check if the column exists in the dataframe

        :param col: The column to check
        :type col: dataframe column
        :return: return true if the column exists
        :rtype: Boolean
        """
        if col in self.df.columns:
            return True
        else:
            return False

    def add_array_dimensions(self):
        """Add ni, nj and nk columns with the help of the d column in the dataframe 

        :raises ArrayIs3dError: the array present in the d column is 3D
        :return: df
        :rtype: pd.DataFrame
        """
        for row in self.df.itertuples():
            array = row.d
            a = np.array([[float(j) for j in i.split(',')] for i in array.split(';')], dtype=np.float32, order='F')
            if(a.ndim == 1):
                ni = np.shape(a)[0]
                nj = 0
                nk = 1

            if(a.ndim == 2):
                ni = np.shape(a)[0]
                nj = np.shape(a)[1]
                nk = 1

            if(a.ndim == 3):
                raise ArrayIs3dError('The numpy array you created from the string array is 3D and it should not be 3d')
            self.df.at[row.Index, "ni"] = ni
            self.df.at[row.Index, "nj"] = nj
            self.df.at[row.Index, "nk"] = nk
        return self.df

    def add_nbits(self):
        """Add the nbits column in the dataframe with a default value of 24
        """
        if(not self.column_exists("nbits")):
            self.df["nbits"] = NBITS_DEFAULT

    def add_datyp(self):
        """Add the datyp column in the dataframe with a default value of 1
        """
        if(not self.column_exists("datyp")):
            self.df["datyp"] = DATYP_DEFAULT

    def add_grtyp(self):
        """Add the grtyp column in the dataframe with a default value of X
        """
        if(not self.column_exists("grtyp")):
            self.df["grtyp"] = GRTYP_DEFAULT

    def add_typvar(self):
        """Add the typvar column in the dataframe with a default value of X
        """
        if(not self.column_exists("typvar")):
            self.df["typvar"] = TYPVAR_DEFAULT

    def add_date(self):
        """Add dateo and datev columns in the dataframe with default values of encoded utcnow
        """
        dateo_encoded = create_encoded_dateo(datetime.datetime.utcnow())
        if(not self.column_exists("dateo") and not self.column_exists("datev")):
            self.df["dateo"] = dateo_encoded
            self.df["datev"] = self.df["dateo"]

    def add_ip2_ip3(self):
        """Add ip2 and ip3 columns in the dataframe with a default value of 0
        """
        if(not self.column_exists("ip2")):
            self.df["ip2"] = IP2_DEFAULT
        if(not self.column_exists("ip3")):
            self.df["ip3"] = IP3_DEFAULT

    def add_ig(self):
        """Add ig1, ig2, ig3, ig4 columns in the dataframe with a default value of 0
        """
        if(not self.column_exists("ig1")):
            self.df["ig1"] = IG1_DEFAULT

        if(not self.column_exists("ig2")):
            self.df["ig2"] = IG2_DEFAULT

        if(not self.column_exists("ig3")):
            self.df["ig3"] = IG3_DEFAULT

        if(not self.column_exists("ig4")):
            self.df["ig4"] = IG4_DEFAULT

    def add_etiket(self):
        """Add the etiket column in the dataframe with a default value of CSVREADER
        """
        if(not self.column_exists("etiket")):
            self.df["eticket"] = ETIKET_DEFAULT

    def add_ip1(self):
        """Add the ip1 column with the help of the level column. 
        The level column is deleted after the data been encoded and put on the ip1 column

        :raises Ip1andLevelExistsError: ip1 and level column exists in the given dataframe
        """
        if self.column_exists("level") and (not self.column_exists("ip1")) and self.encode_ip1:
            for row in self.df.itertuples():
                level = float(row.level)
                ip1 = create_encoded_ip1(level=level, ip1_kind=IP1_KIND, mode=rmn.CONVIP_ENCODE)
                self.df.at[row.Index, "ip1"] = ip1

        elif self.column_exists("level") and (not self.column_exists("ip1")) and (not self.encode_ip1):
            for row in self.df.itertuples():
                level = float(row.level)
                ip1 = level
                self.df.at[row.Index, "ip1"] = ip1

        elif (self.column_exists("ip1")) and (self.column_exists("level")):
            raise Ip1andLevelExistsError("IP1 AND LEVEL EXISTS IN THE CSV FILE")

        # Remove level after we added ip1 column
        self.df.drop(columns=["level"], inplace=True, errors="ignore")

    def add_deet(self):
        """Add a column deet in the dataframe with a default value of 0
        """
        if(not self.column_exists("deet")):
            self.df["deet"] = DEET_DEFAULT

    def add_npas(self):
        """Add a column npas in the dataframe with a default value of 0
        """
        if(not self.column_exists("npas")):
            self.df["npas"] = NPAS_DEFAULT

    def check_array_dimensions(self):
        """Check if etiket and name is the same as the previous row to compare dimensions

        :raises DimensionError: the array with the same var and etiket dont have the same dimension
        """
        groups = self.df.groupby(['nomvar', 'typvar', 'etiket', 'dateo', 'ip2', 'ip3', 'deet', 'npas', 'datyp',
                                  'nbits', 'ig1', 'ig2', 'ig3', 'ig4'])

        for _, df in groups:
            if df.ni.unique().size != 1:
                raise DimensionError("Array with the same nomvar and etiket dont have the same dimension ")
            if df.nj.unique().size != 1:
                raise DimensionError("Array with the same nomvar and etiket dont have the same dimension ")

    def to_numpy_array(self):
        """Takes a string array and transform it to a numpy array"""
        array_list = []
        for i in self.df.index:
            a = CsvArray(self.df.at[i, "d"])
            a = a.to_numpy()
            array_list.append(a)
        self.df["d"] = array_list

    def change_column_dtypes(self):
        """Change the columns types to the correct types in the dataframe
        """
        self.df = self.df.astype({'ni': 'int32', 'nj': 'int32', 'nk': 'int32', 'nomvar': "str", 'typvar': 'str', 'etiket': 'str',
                                  'dateo': 'int32', 'ip1': 'int32', 'ip2': 'int32', 'ip3': 'int32', 'datyp': 'int32', 'nbits': 'int32',
                                  'ig1': 'int32', 'ig2': 'int32', 'ig3': 'int32', 'ig4': 'int32', 'deet': 'int32', 'npas': 'int32',
                                  'grtyp': 'str', 'datev': 'int32'})

