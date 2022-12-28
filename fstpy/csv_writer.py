from .utils import CsvArray
from fstpy.std_reader import compute
import os.path
import pandas as pd

BASE_COLUMNS = ['nomvar', 'typvar', 'etiket', 'ni', 'nj', 'nk', 'dateo', 'ip1', 'ip2', 'ip3', 'deet', 'npas', 'datyp', 'nbits', 'grtyp', 'ig1', 'ig2', 'ig3', 'ig4', 'datev','d']

class CsvFileWriterError(Exception):
    pass

class CsvFileWriter:
    """"Writes a csv file from the dataframe given.

    - The dataframe given has to have all these columns or the writer won't work:

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
        | d       | str    |                data                                  | 
        +---------+--------+------------------------------------------------------+


    :param path: path of the csv file created
    :type path: str
    :param df: dataframe
    :type df: pd.DataFrame
    :param overwrite: overwrite if the file in the path specified already exists, defaults to False
    :type overwrite: bool, optional
    """
    
    def __init__(self,path: str,df: pd.DataFrame,overwrite=False): 
        self.path = path
        self.df = df
        self.overwrite = overwrite

    def to_csv(self):
        """Create a new csv file that display the information in the dataframe

        :raises CsvFileWriterError: The path already has a file. Overwrite default value is false.
        """
        if not os.path.isfile(self.path) or self.overwrite == True:
            self.df = compute(self.df)
            self.convert_d_column()
            self.remove_grid_column()
            self.check_columns()
            self.change_column_dtypes()
            self.df.to_csv(self.path,index=False)
        else:
            raise CsvFileWriterError("The file created already exists in the path specified. Use overwrite flag to avoid this error")

    def remove_columns(self):
        """Remove columns that are not part of the necessary columns in the csv file
        """

        if len(self.df.columns) > len(BASE_COLUMNS):
            diff = list(set(BASE_COLUMNS) ^ set(self.df.columns))
            for i in diff:
                self.df.drop(str(i), inplace=True, axis=1)
            

    def change_column_dtypes(self):
        """Change the columns types to the correct types in the dataframe
        """

        self.df = self.df.astype({'ni': 'int32', 'nj': 'int32', 'nk': 'int32', 'nomvar': "str", 'typvar': 'str', 'etiket': 'str',
                                  'dateo': 'int32', 'ip1': 'int32', 'ip2': 'int32', 'ip3': 'int32', 'datyp': 'int32', 'nbits': 'int32',
                                  'ig1': 'int32', 'ig2': 'int32', 'ig3': 'int32', 'ig4': 'int32', 'deet': 'int32', 'npas': 'int32',
                                  'grtyp': 'str', 'datev': 'int32'})

    def remove_grid_column(self):
        """Remove the grid column ,because it's added in the csv reader"""
        self.df.drop('grid', inplace=True, axis=1)

    def check_columns(self):
        """Check that all the columns in the dataframe are the correct ones. If any of the columns are not supposed 
        to be there they are deleted

        :return: True
        :rtype: Boolean
        """

        all_the_cols = BASE_COLUMNS
        all_the_cols.sort()
        list_of_hdr_names = self.df.columns.tolist()
        list_of_hdr_names.sort()
        if all_the_cols == list_of_hdr_names:
            return True
        else:
            self.remove_columns()
            

    def convert_d_column(self):
        """Convert the d column to a string in the right format"""

        for i in self.df.index:
            self.df.at[i,'d'] = CsvArray(self.df.at[i,'d']).to_str()
        


    


        