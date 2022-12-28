# -*- coding: utf-8 -*-
import inspect
import logging
import os
from functools import wraps

import dask.array as da
import numpy as np
import rpnpy.librmn.all as rmn


# from io import StringIO 
# import sys
#
# class StdoutCapture(list):
#     def __enter__(self):
#         self._stdout = sys.stdout
#         sys.stdout = self._stringio = StringIO()
#         return self
#     def __exit__(self, *args):
#         self.extend(self._stringio.getvalue().splitlines())
#         del self._stringio    # free up some memory
#         sys.stdout = self._stdout
        
def initializer(func):
    """
    Automatically assigns the parameters.

    >>> class process:
    ...     @initializer
    ...     def __init__(self, cmd, reachable=False, user='root'):
    ...         pass
    >>> p = process('halt', True)
    >>> p.cmd, p.reachable, p.user
    ('halt', True, 'root')
    """
    names, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations = inspect.getfullargspec(
        func)
    #names, varargs, keywords, defaults = inspect.getfullargspec(func)

    @wraps(func)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        for name, default in zip(reversed(names), reversed(defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)

        func(self, *args, **kargs)

    return wrapper


def delete_file(my_file: str):
    """delete a file by path

    :param my_file: path to file to delete
    :type my_file: str
    """
    import os
    if os.path.exists(my_file):
        os.unlink(my_file)


def get_file_list(pattern: str) -> str:
    """Gets the list of files for provided path expression with wildcards

    :param pattern: a directory with regex pattern to find files in
    :type pattern: str
    :return: a list of files
    :rtype: str
    """
    import glob
    files = glob.glob(pattern)
    return files


def ip_from_value_and_kind(value: float, kind: str) -> int:
    """Create an encoded ip value out of a value (float) and a printable kind.
    Valid kinds are m,sg,M,hy,th,H and mp

    :param level: a level value as a float
    :type level: float
    :param kind: a textual representation of kind, m,sg,M,hy,th,H or mp
    :type kind: str
    :return: encoded ip value
    :rtype: int
    """
    d = {
        'm': 0,
        'sg': 1,
        'mb': 2,
        'M': 4,
        'hy': 5,
        'th': 6,
        'H': 10,
        'mp': 21
    }

    pk = rmn.listToFLOATIP((value, value, d[kind.strip()]))

    if kind.strip() == 'H':
        (_, ip, _) = rmn.convertPKtoIP(
            rmn.listToFLOATIP((value, value, d["m"])), pk, pk)
    else:
        (ip, _, _) = rmn.convertPKtoIP(pk, pk, pk)
    return ip


def column_descriptions():
    """Prints the base attributes descriptions
    """
    logging.info('nomvar: variable name')
    logging.info(
        'typvar: type of field ([F]orecast, [A]nalysis, [C]limatology)')
    logging.info(
        'etiket: concatenation of label, run, implementation and ensemble_member')
    logging.info('ni: first dimension of the data field - relates to shape')
    logging.info('nj: second dimension of the data field - relates to shape')
    logging.info('nk: third dimension of the data field - relates to shape')
    logging.info('dateo: date of observation time stamp')
    logging.info('ip1: encoded vertical level')
    logging.info(
        'ip2: encoded forecast hour, but can be used in other ways by encoding an ip value')
    logging.info('ip3: user defined identifier')
    logging.info(
        'deet: length of a time step in seconds - usually invariable - relates to model ouput times')
    logging.info('npas: time step number')
    logging.info('datyp: data type of the elements (int,float,str,etc)')
    logging.info(
        'nbits: number of bits kept for the elements of the field (16,32,etc)')
    logging.info(
        'ig1: first grid descriptor, helps to associate >>, ^^, !!, HY, etc with variables')
    logging.info(
        'ig2: second grid descriptor, helps to associate >>, ^^, !!, HY, etc with variables')
    logging.info(
        'ig3: third grid descriptor, helps to associate >>, ^^, !!, HY, etc with variables')
    logging.info(
        'ig4: fourth grid descriptor, helps to associate >>, ^^, !!, HY, etc with variables')
    logging.info(
        'grtyp: type of geographical projection identifier (Z, X, Y, etc)')
    logging.info(
        'datev: date of validity (dateo + deet * npas) Will be set to -1 if dateo invalid')
    logging.info(
        'd: data associated to record, empty until data is loaded - either a numpy array or a daks array for one level of data')
    logging.info(
        'key: key/handle of the record - used by rpnpy to locate records in a file')
    logging.info(
        'shape: (ni, nj, nk) dimensions of the data field - an attribute of the numpy/dask array (array.shape)')


def get_num_rows_for_reading(df):
    max_num_rows = 128
    num_rows = os.getenv('FSTPY_NUM_ROWS')

    if num_rows is None:
        num_rows = max_num_rows
    else:
        num_rows = int(num_rows)    
    num_rows = min(num_rows,len(df.index))    
    return num_rows

class ConversionError(Exception):
    pass

def to_numpy(arr: "np.ndarray|da.core.Array") -> np.ndarray:
    """If the array is of numpy type, no op, else compute de daks array to get a numpy array

    :param arr: array to convert
    :type arr: np.ndarray|da.core.Array
    :raises ConversionError: Raised if not a numpy or dask array
    :return: a numpy array
    :rtype: np.ndarray
    """
    if arr is None:
        return arr
    if isinstance(arr, da.core.Array):   
        return arr.compute()  
    elif isinstance(arr,np.ndarray):    
        return arr    
    else:
        raise ConversionError('to_numpy - Array is not an array of type numpy or dask')    

def to_dask(arr:"np.ndarray|da.core.Array") -> da.core.Array:
    """If the array is of dask type, no op, else comvert array to dask array

    :param arr: array to convert
    :type arr: np.ndarray|da.core.Array
    :raises ConversionError: Raised if not a numpy or dask array
    :return: a dask array
    :rtype: da.core.Array
    """
    if arr is None:
        return arr
    if isinstance(arr, da.core.Array):   
        return arr
    elif isinstance(arr, np.ndarray):   
        return da.from_array(arr).astype(np.float32)        
    else:    
        raise ConversionError('to_dask - Array is not an array of type numpy or dask')    




class FstPrecision:

    datyp_priority = {-1:-1,0:0,1:5,2:1,4:3,5:7,6:4,7:0,8:9,130:2,133:8,134:6}

    def __init__(self, datyp:int, nbits:int):
        self.nbits = nbits
        self.datyp = self.datyp_priority[datyp]    


    def max(self,other):
        nbits = self.nbits if self.nbits >= other.nbits else other.nbits
        datyp = self.datyp if self.datyp >= other.datyp else other.datyp
        return datyp,nbits


        # // Help find the most precise DATA_TYPE to store value, compressed is prefered for same DATA_TYPE.
        # // The bigger the returned value is, the better the DATA_TYPE is.
        # static int getDataTypePrecisionRanking(DATA_TYPE_T dataType)
        # {
        #     switch (dataType)
        #     {
        #     case DATA_TYPE_NOT_SET:
        #         return -1;
        #     case STRING:
        #         return 0; // should not be use to store value
        #     case RAW:
        #         return 0; // not sure maybe it's the one that will hold the best precision??
        #     case UNSIGNED:
        #         return 1;
        #     case COMPRESSED_UNSIGNED:
        #         return 2;
        #     case INTEGER:
        #         return 3;
        #     case FLOAT_FOR_COMPRESSOR:
        #         return 4;
        #     case FLOAT:
        #         return 5;
        #     case COMPRESSED_FLOAT:
        #         return 6;
        #     case IEEE:
        #         return 7;
        #     case COMPRESSED_IEEE:
        #         return 8;
        #     case COMPLEX_IEEE:
        #         return 9;

# Decorator for efficiently converting a scalar function to a vectorized
# function.
def vectorize (f, otypes=None):
    from functools import wraps
    import numpy as np
    @wraps(f)
    def vectorized_f (*x):
        from pandas import Series, unique
        n = max(len(y) if hasattr(y,'__len__') and not isinstance(y,str) else 1 for y in x)
        # Degenerate case: input vector has length 0.
        if n == 0:
            if otypes is None:
                return []
            else:
                return ([],)*len(otypes)
        # Expand any scalar arguments.
        x = [y if hasattr(y,'__len__') and not isinstance(y,str) else (y,)*n for y in x]
        # Get unique values
        x = list(zip(*x))
        inputs = unique(x)
        outputs = list(map(f,*zip(*inputs)))
        table = dict(zip(inputs,outputs))
        result = Series(x).map(table)
        # Multiple outputs?
        if isinstance(outputs[0],tuple):
            result = list(zip(*result.values))
        else:
            result = [result.values]
        if otypes is None:
            result = tuple(np.array(r) for r in result)
        else:
            result = tuple(np.array(r,dtype=o) for r,o in zip(result,otypes))
        if len(result) == 1: result = result[0]  # Only one output
        return result
    return vectorized_f
# In case of emergency, break glass.
#from numpy import vectorize

class ArrayIsNotNumpyStrError(Exception):
    pass


class ArrayIs3dError(Exception):
    pass


class ArrayIsNotStringOrNp(Exception):
    pass

class CsvArray:
    """A class that represents a csv formatted array

    :param array: An array with the data
    :type array: string or numpy array
    :raises ArrayIsNotStringOrNp: the array is not formed from strings or numpy array
    """

    def __init__(self, array):
        self.array = array
        if(self.validate_array()):
            pass
        else:
            raise ArrayIsNotStringOrNp("The array is not a string or a numpy aray")

    def validate_array(self):
        """Validate that the array is either a string or a numpy array

        :raises ArrayIs3dError: the array provided is 3D
        :rtype: Boolean
        """

        if(type(self.array) == np.ndarray or type(self.array) == str):
            return True
        else:
            return False

    def to_numpy(self):
        """Transform self.array to a numpy array

        :raises ArrayIs3dError: the array provided is 3D
        :return: numpy array
        """
        if isinstance(self.array, str):
            b = self.array
            a = np.array([[float(j) for j in i.split(',')] for i in b.split(';')], dtype=np.float32, order='F')
            if(a.ndim == 3):
                raise ArrayIs3dError('The numpy array you created from the string array is 3D and it should not be 3d')
            return a
        else:
            return self.array

    def to_str(self):
        """Transform numpy array to a string

        :raises ArrayIs3dError: the array provided is 3D
        :return: string array
        """
        if isinstance(self.array, np.ndarray):
            b = self.array
            dim0 = []
            ndim0 = b.shape[0]

            for i in range(ndim0):
                dim0.append([b[i, j] for j in range(b.shape[1])])

            dim0 = []
            ndim0 = b.shape[0]

            for i in range(ndim0):
                dim0.append([b[i, j] for j in range(b.shape[1])])

            s = ""

            for i in range(ndim0):
                s1 = str(dim0[i]).replace("[", "")
                s1 = s1.replace("]", ";")
                s += s1
            s = s.replace(" ", "")
            s = s.rstrip(s[-1])
            return s
        else:
            return self.array