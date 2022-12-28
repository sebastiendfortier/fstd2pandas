# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import copy
import dask.array as da
from enum import Enum
import logging
import math
import numpy as np
import pandas as pd
from typing import Tuple, Final
import rpnpy.librmn.all as rmn
import rpnpy.vgd.all as vgd

STANDARD_ATMOSPHERE: Final[float] = 1013.25
"""Standard Atmosphere constant"""

###############################################################
#####             Vertical Coordinate Enum                #####
###############################################################
class VerticalCoordType(Enum):
    """Enum for vertical coordinate types

    :param Enum: coordinate name, value pairs
    :type Enum: Enum
    """
    SIGMA_1001: Final[int] = 1001
    ETA_1002: Final[int] = 1002
    HYBRID_NORMALIZED_1003: Final[int] = 1003
    PRESSURE_2001: Final[int] = 2001
    HYBRID_5001: Final[int] = 5001
    HYBRID_5002: Final[int] = 5002
    HYBRID_5003: Final[int] = 5003
    HYBRID_5004: Final[int] = 5004
    HYBRID_5005: Final[int] = 5005
    METER_SEA_LEVEL: Final[int] = 0
    METER_GROUND_LEVEL: Final[int] = 4
    UNKNOWN: Final[int] = 9999

    def __lt__(self, other):
        """Defined for grouping in dataframe

        :param other: enum to compare to
        :type other: Enum
        :return: True if other is greater than instance
        :rtype: bool
        """
        return self.value < other.value

    def __str__(self):
        return self.name


vctype_dict = {
    'SIGMA_1001': VerticalCoordType.SIGMA_1001,
    'ETA_1002': VerticalCoordType.ETA_1002,
    'HYBRID_NORMALIZED_1003': VerticalCoordType.HYBRID_NORMALIZED_1003,
    "PRESSURE_2001": VerticalCoordType.PRESSURE_2001,
    "HYBRID_5001": VerticalCoordType.HYBRID_5001,
    "HYBRID_5002": VerticalCoordType.HYBRID_5002,
    "HYBRID_5003": VerticalCoordType.HYBRID_5003,
    "HYBRID_5004": VerticalCoordType.HYBRID_5004,
    "HYBRID_5005": VerticalCoordType.HYBRID_5005,
    "METER_SEA_LEVEL": VerticalCoordType.METER_SEA_LEVEL,
    "METER_GROUND_LEVEL": VerticalCoordType.METER_GROUND_LEVEL,
    "UNKNOWN": VerticalCoordType.UNKNOWN,
}
"""Dictionnary for string Enum correspondance"""


class VerticalCoordError(Exception):
    pass


###############################################################
#####             Vertical Coordinate Classes             #####
###############################################################
class VerticalCoord(ABC):
    """Super class for Vertical coordinate types

    :param file_df: Dataframe containing fields of only one file (single path)
    :type file_df: pd.DataFrame
    :param meta_df: Dataframe of all the vertical coordinate meta data for a single grid in a file
    :type meta_df: pd.DataFrame
    :param df: Dataframe all the fields of a single grid and single date of validity from a single file
    :type df: pd.DataFrame
    :param vcode: vcode of the vertical coordinate type, set by sub-class, defaults to 0
    :type vcode: int, optional
    :param kind: kind of the vertical coordinate type, set by sub-class, defaults to 0
    :type kind: int, optional
    :param version: version of the vertical coordinate type, set by sub-class, defaults to 0
    :type version: int, optional
    """

    def __init__(self, file_df: pd.DataFrame, meta_df: pd.DataFrame, df: pd.DataFrame, vcode=0, kind=0, version=0) -> None:
        self.vcode = vcode
        self.kind = kind
        self.version = version
        self.file_df = file_df
        self.meta_df = meta_df
        self.df = df
        self.check_dataframes()
        self.get_levels_and_ips_df()
        self.get_toctoc()
        self.get_p0()
        self.get_pt()
        self.get_hy()

    def __repr__(self):
        """repr of the class VerticalCoord"""
        return f"""vcode: {self.vcode}
        lvl_ip_df\n{self.lvl_ip_df.head()}
        df\n{self.df[['nomvar','ip1','datev','vctype']].head().to_string()}
        hy_df{self.hy_df[['nomvar','ip1']].to_string() if not self.hy_df.empty else ': no hy'}
        p0_df{self.p0_df[['nomvar','ip1','datev','datyp','nbits']].to_string() if not self.p0_df.empty else ': no p0'}
        pt_df{self.pt_df[['nomvar','ip1','datev','vctype']].to_string() if not self.pt_df.empty else ': no pt'}
        toctoc_df{self.toctoc_df[['nomvar','ip1','datev']].to_string() if not self.toctoc_df.empty else ': no !!'}"""

    def check_dataframes(self):
        """Checks that the different dataframes containe the appropriate columns

        :raises VerticalCoordError: more than one path in df
        :raises VerticalCoordError: more than one vctype in df
        :raises VerticalCoordError: more than one datev in df
        :raises VerticalCoordError: more than one grid in meta_df
        :raises VerticalCoordError: more than one path in file_df
        """
        from .dataframe import add_path_and_key_columns, add_columns
        # one path
        # one vctype
        # one datev
        if 'path' not in self.df.columns:
            self.df = add_path_and_key_columns(self.df)

        if len(list(self.df.path.unique())) > 1:
            raise VerticalCoordError('More than one path in dataframe, cannot proceed!')

        if 'vctype' not in self.df.columns:
            self.df = add_columns(self.df, 'ip_info')

        if len(list(self.df.vctype.unique())) > 1:
            raise VerticalCoordError('More than one vctype in dataframe, cannot proceed!')

        if len(list(self.df.datev.unique())) > 1:
            raise VerticalCoordError('More than one datev in dataframe, cannot proceed!')

        if len(list(self.meta_df.grid.unique())) > 1:
            raise VerticalCoordError('More than one grid in meta data dataframe, cannot proceed!')

        if len(list(self.df.grid.unique())) > 1:
            raise VerticalCoordError('More than one grid in dataframe, cannot proceed!')

        if len(list(self.file_df.path.unique())) > 1:
            raise VerticalCoordError("file_df contains more than one path, cannot proceed!")

    def get_hy(self):
        """Try and get Hy field from the file_df"""
        from .std_io import add_dask_column
        self.hy_df = self.file_df.loc[(self.file_df.nomvar == "HY")]
        if not self.hy_df.empty:
            if 'd' not in self.hy_df.columns:
                self.hy_df = add_dask_column(self.hy_df)

    def get_p0(self):
        """Try and get P0 field from the meta_df"""
        from .std_io import add_dask_column
        self.p0_df = self.meta_df.loc[(self.meta_df.nomvar == "P0") & (self.meta_df.datev == self.df.datev.unique()[0])]
        if not self.p0_df.empty:
            if 'd' not in self.p0_df.columns:
                self.p0_df = add_dask_column(self.p0_df)

    def get_pt(self):
        """Try and get PT field from the meta_df"""
        from .std_io import add_dask_column
        self.pt_df = self.meta_df.loc[(self.meta_df.nomvar == "PT") & (self.meta_df.datev == self.df.datev.unique()[0])]
        if not self.pt_df.empty:
            if 'd' not in self.pt_df.columns:
                self.pt_df = add_dask_column(self.pt_df)

    def get_toctoc(self):
        """Try and get !! field from the meta_df"""
        from .std_io import add_dask_column
        self.toctoc_df = self.meta_df.loc[(self.meta_df.nomvar == "!!") & (self.meta_df.ig1 == self.vcode)]
        if not self.toctoc_df.empty:
            if 'd' not in self.toctoc_df.columns:
                self.toctoc_df = add_dask_column(self.toctoc_df)

    def get_levels_and_ips_df(self):
        """Creates a dataframe of levels and corresponding ip1's"""
        def getLevel(ip):
            (level, _) = rmn.convertIp(rmn.CONVIP_DECODE, int(ip))
            return level
        self.ips = list(self.df.ip1.unique())
        self.newstyle = np.where(self.df.ip1.unique() > 32767, True, False).all()
        self.levels = [getLevel(ip) for ip in self.ips]
        self.lvl_ip_df = pd.DataFrame({'level': self.levels, 'ip1': self.ips})

    @abstractmethod
    def pressure(self):
        raise NotImplementedError("You should implement this")

    @abstractmethod
    def pressure_standard_atmosphere(self):
        raise NotImplementedError("You should implement this")

    def get_px_precision(self) -> 'Tuple(int,int)':
        """Generic method to get the presision (nbits,datyp) of the px field

        :return: nbits and datyp from p0 field
        :rtype: Tuple(int,int)
        """
        nbits = self.p0_df.iloc[0].nbits
        datyp = self.p0_df.iloc[0].datyp
        return nbits, datyp

    def create_result_container(self, nomvar: str, etiket: str, unit: str, description: str) -> pd.DataFrame:
        """Creates a DataFrame of PX to hold results, with appropriate number of levels

        :param nomvar: nomvar to assign
        :type nomvar: str
        :param etiket: etiket to assign
        :type etiket: str
        :param unit: unit to assign
        :type unit: str
        :param description: description to assign
        :type description: str
        :return: DataFrame of PX to hold results
        :rtype: pd.DataFrame
        """
        base_dict = self.df.iloc[0].to_dict()
        res_df = pd.DataFrame([base_dict for l in self.levels])
        nbits, datyp = self.get_px_precision()
        res_df['nomvar'] = nomvar
        res_df['etiket'] = etiket
        res_df['unit'] = unit
        res_df['nbits'] = nbits
        res_df['datyp'] = datyp
        res_df['description'] = description
        res_df['ip1'] = [int(self.lvl_ip_df.loc[self.lvl_ip_df.level == lvl].iloc[0].ip1) for lvl in self.levels]
        res_df['level'] = self.levels
        return res_df

    def create_px_container(self) -> pd.DataFrame:
        """Specific wrapper for PX dataframe container

        :return: DataFrame of PX to hold results
        :rtype: pd.DataFrame
        """
        res_df = self.create_result_container('PX', 'PRESSR', 'hectoPascal', 'Pressure of the Model')
        return res_df

    def create_pxsa_container(self) -> pd.DataFrame:
        """Specific wrapper for PXSA dataframe container

        :return: DataFrame of PXSA to hold results
        :rtype: pd.DataFrame
        """
        res_df = self.create_result_container('PXSA', 'PRESSR', 'millibar', 'Pressure of the model standard atmosphere')
        return res_df

###############################################################
#####          Vertical Coordinate Hybrid Class           #####
###############################################################
class VerticalCoordHybrid(VerticalCoord):
    """Specific constructor for 5002 vertical coordinate type"""

    def __init__(self, file_df: pd.DataFrame, meta_df: pd.DataFrame, df: pd.DataFrame, vcode: int, kind: int, version: int) -> None:
        super().__init__(file_df, meta_df, df, vcode, kind, version)
        if self.p0_df.empty:
            raise VerticalCoordError('Missing p0, cannot proceed!')
        if self.toctoc_df.empty:
            raise VerticalCoordError('Missing !!, cannot proceed!')

    def get_px_precision(self):
        return super().get_px_precision()

    def get_vcoord_table(self):
        """Creates a dataframe of ip1's and corresponding A and B values"""
        toctoc = self.toctoc_df.iloc[0]['d']
        self.pref = toctoc[1][1]
        vcoord_table = pd.DataFrame({'ip1': toctoc[0].astype(np.int32), 'A': toctoc[1], 'B': toctoc[2]})
        self.vcoord_table = pd.merge(vcoord_table, self.lvl_ip_df, on="ip1")
        self.vcoord_table.drop_duplicates(inplace=True, ignore_index=True)

    def hybrid_pressure_with_toctoc(self) -> 'list(da.Array)':
        """Calculates the pressure for all levels of variables in df

        :return: pressure array for every level
        :rtype: list(da.Array)
        """
        self.get_vcoord_table()
        self.levels = list(self.vcoord_table.level)
        p0 = self.p0_df.iloc[0].d
        s = np.log(p0*100./self.pref)
        pres = [(np.exp(self.vcoord_table.at[idx, 'A'] + self.vcoord_table.at[idx, 'B'] * s) /
                 100.0).astype(np.float32) for idx in self.vcoord_table.index]
        return pres

    def hybrid_pressure_with_toctoc_sa(self) -> 'list(da.Array)':
        """Calculates the standard atmosphere pressure for all levels of variables in df

        :return: standard atmosphere pressure array for every level
        :rtype: list(da.Array)
        """
        self.get_vcoord_table()
        p0 = self.p0_df.iloc[0].d
        self.vcoord_table['std_atm_value'] = np.exp(
            self.vcoord_table['A'] + self.vcoord_table['B'] * math.log(STANDARD_ATMOSPHERE * 100.0 / self.pref)) / 100.0
        pres = [da.full(p0.shape, self.vcoord_table.at[idx, 'std_atm_value'], dtype=np.float32, order='F')
                for idx in self.vcoord_table.index]
        return pres    

    def pressure(self):
        pres = self.hybrid_pressure_with_toctoc()
        res_df = super().create_px_container()
        res_df['d'] = pres
        return res_df

    def pressure_standard_atmosphere(self):
        pres = self.hybrid_pressure_with_toctoc_sa()
        res_df = super().create_pxsa_container()
        res_df['d'] = pres
        return res_df


###############################################################
#####          1001 Vertical Coordinate Class             #####
###############################################################
class VerticalCoord1001(VerticalCoord):
    """Specific constructor for 1001 vertical coordinate type"""

    def __init__(self, file_df: pd.DataFrame, meta_df: pd.DataFrame, df: pd.DataFrame) -> None:
        super().__init__(file_df, meta_df, df, 1001, 1, 1)
        if self.p0_df.empty:
            raise VerticalCoordError('Missing p0, cannot proceed!')

    def get_px_precision(self):
        return super().get_px_precision()

    def pressure(self):
        p0 = self.p0_df.iloc[0].d
        pres = [lvl * p0 for lvl in self.levels]
        res_df = super().create_px_container()
        res_df['d'] = pres
        return res_df

    def pressure_standard_atmosphere(self):
        p0 = self.p0_df.iloc[0].d
        pres = [da.full(p0.shape, STANDARD_ATMOSPHERE * level, dtype=np.float32, order='F') for level in self.levels]
        res_df = super().create_pxsa_container()
        res_df['d'] = pres
        return res_df

###############################################################
#####          1002 Vertical Coordinate Class             #####
###############################################################
class VerticalCoord1002(VerticalCoord):
    """Specific constructor for 1002 vertical coordinate type"""

    def __init__(self, file_df: pd.DataFrame, meta_df: pd.DataFrame, df: pd.DataFrame) -> None:
        super().__init__(file_df, meta_df, df, 1002, 1, 2)
        if self.pt_df.empty:
            raise VerticalCoordError('Missing pt, cannot proceed!')
        if self.p0_df.empty:
            raise VerticalCoordError('Missing p0, cannot proceed!')

    def get_px_precision(self):
        return super().get_px_precision()

    def pressure(self):
        p0 = self.p0_df.iloc[0].d
        pt = self.pt_df.iloc[0].d
        pres = [lvl * (p0-pt) + pt for lvl in self.levels]
        res_df = super().create_px_container()
        res_df['d'] = pres
        return res_df

    def pressure_standard_atmosphere(self):
        pt = self.pt_df.iloc[0].d
        ptop = pt  # / 100.0
        pres = [(ptop * (1.0 - level)) + level * STANDARD_ATMOSPHERE for level in self.levels]
        res_df = super().create_pxsa_container()
        res_df['d'] = pres
        return res_df

###############################################################
#####          2001 Vertical Coordinate Class             #####
###############################################################
class VerticalCoord2001(VerticalCoord):
    """Specific constructor for 2001 vertical coordinate type"""

    def __init__(self, file_df: pd.DataFrame, meta_df: pd.DataFrame, df: pd.DataFrame) -> None:
        super().__init__(file_df, meta_df, df, 2001, 2, 1)

    def get_array_shape(self):
        from .std_io import add_dask_column
        if 'd' not in self.df.columns:
            a_df = pd.DataFrame([self.df.iloc[0].to_dict()])
            a_df = add_dask_column(a_df)
            shape = a_df.iloc[0]['d'].shape
        else:
            shape = self.df.iloc[0]['d'].shape
        return shape

    def get_px_precision(self):
        return 32, 5

    def compute_pressure(self):
        shape = self.get_array_shape()
        pres = [da.full(shape, lvl, dtype=np.float32, order='F') for lvl in self.levels]
        return pres

    def pressure(self):
        pres = self.compute_pressure()
        res_df = super().create_px_container()
        res_df['d'] = pres
        return res_df

    def pressure_standard_atmosphere(self):
        pres = self.compute_pressure()
        res_df = super().create_pxsa_container()
        res_df['d'] = pres
        return res_df

###############################################################
#####          5001 Vertical Coordinate Class             #####
###############################################################
class VerticalCoord5001(VerticalCoord):
    """Specific constructor for 5001 vertical coordinate type"""

    def __init__(self, file_df: pd.DataFrame, meta_df: pd.DataFrame, df: pd.DataFrame) -> None:
        super().__init__(file_df, meta_df, df, 5001, 5, 1)
        if self.hy_df.empty:
            raise VerticalCoordError('Missing hy, cannot proceed!')
        if self.p0_df.empty:
            raise VerticalCoordError('Missing p0, cannot proceed!')

    def get_px_precision(self):
        return super().get_px_precision()

    def get_ptop_pref_rcoef(self):
        from .utils import to_numpy
        hy = to_numpy(self.hy_df.iloc[0]['d'])
        self.ptop = hy[0]
        self.pref = self.hy_df.iloc[0]['ig1']
        self.rcoef = self.hy_df.iloc[0]['ig2'] / 1000.0

    def pressure(self):
        res_df = super().create_px_container()
        p0 = self.p0_df.iloc[0].d
        self.get_ptop_pref_rcoef()
        etatop = self.ptop/self.pref
        B = ((self.levels - etatop) / (1 - etatop)) ** self.rcoef
        A = self.pref * (self.levels - B)
        vcoord_table = pd.DataFrame({'level': self.levels, 'A': A, 'B': B})
        vcoord_table.drop_duplicates(inplace=True, ignore_index=True)
        pres = [(vcoord_table.at[idx, 'A'] + vcoord_table.at[idx, 'B'] * p0).astype(np.float32)
                for idx in vcoord_table.index]
        self.levels = list(vcoord_table.level)
        res_df = super().create_px_container()
        res_df['d'] = pres
        return res_df

    def pressure_standard_atmosphere(self):
        self.get_ptop_pref_rcoef()
        p0 = self.p0_df.iloc[0].d
        term0 = (self.ptop / self.pref)
        term3 = (1.0 / (1.0 - term0))
        term4 = (self.levels - term0)
        evalTerm1 = np.where(term4 < 0, 0., term4)
        # evalTerm1 = (0.0 if term4 < 0 else term4)
        term6 = (evalTerm1 * term3)**self.rcoef
        pres_value = (self.pref * (self.levels - term6)) + term6 * STANDARD_ATMOSPHERE
        pres = [da.full(p0.shape, pv, dtype=np.float32, order='F') for pv in pres_value]
        res_df = super().create_pxsa_container()
        res_df['d'] = pres
        return res_df

###############################################################
#####          5002 Vertical Coordinate Class             #####
###############################################################
class VerticalCoord5002(VerticalCoordHybrid):
    """Specific constructor for 5002 vertical coordinate type"""

    def __init__(self, file_df: pd.DataFrame, meta_df: pd.DataFrame, df: pd.DataFrame) -> None:
        super().__init__(file_df, meta_df, df, 5002, 5, 2)

###############################################################
#####          5003 Vertical Coordinate Class             #####
###############################################################
class VerticalCoord5003(VerticalCoordHybrid):
    """Specific constructor for 5003 vertical coordinate type"""

    def __init__(self, file_df: pd.DataFrame, meta_df: pd.DataFrame, df: pd.DataFrame) -> None:
        super().__init__(file_df, meta_df, df, 5003, 5, 3)

###############################################################
#####          5004 Vertical Coordinate Class             #####
###############################################################
class VerticalCoord5004(VerticalCoordHybrid):
    """Specific constructor for 5004 vertical coordinate type"""

    def __init__(self, file_df: pd.DataFrame, meta_df: pd.DataFrame, df: pd.DataFrame) -> None:
        super().__init__(file_df, meta_df, df, 5004, 5, 4)

###############################################################
#####          5005 Vertical Coordinate Class             #####
###############################################################
class VerticalCoord5005(VerticalCoordHybrid):
    """Specific constructor for 5005 vertical coordinate type"""

    def __init__(self, file_df: pd.DataFrame, meta_df: pd.DataFrame, df: pd.DataFrame) -> None:
        super().__init__(file_df, meta_df, df, 5005, 5, 5)

###############################################################
#####         Unknown Vertical Coordinate Class           #####
###############################################################
class VerticalCoordUnknown(VerticalCoord):
    """Specific constructor for UNKNOWN vertical coordinate type"""

    def __init__(self, file_df: pd.DataFrame, meta_df: pd.DataFrame, df: pd.DataFrame) -> None:
        super().__init__(file_df, meta_df, df)

    def __str__(self):
        return ''

    def pressure(self):
        logging.warning('Cannot compute pressure!')
        return pd.DataFrame(dtype=object)

    def pressure_standard_atmosphere(self):
        logging.warning('Cannot compute pressure at standard atmosphere!')
        return pd.DataFrame(dtype=object)


vertical_coord = {
    VerticalCoordType.SIGMA_1001: VerticalCoord1001,
    VerticalCoordType.ETA_1002: VerticalCoord1002,
    VerticalCoordType.PRESSURE_2001: VerticalCoord2001,
    VerticalCoordType.HYBRID_5001: VerticalCoord5001,
    VerticalCoordType.HYBRID_5002: VerticalCoord5002,
    VerticalCoordType.HYBRID_5003: VerticalCoord5003,
    VerticalCoordType.HYBRID_5004: VerticalCoord5004,
    VerticalCoordType.HYBRID_5005: VerticalCoord5005,
    VerticalCoordType.UNKNOWN: VerticalCoordUnknown,
}


def get_vertical_coord(file_df: pd.DataFrame, meta_df: pd.DataFrame, df: pd.DataFrame) -> VerticalCoord:
    """Factory function to get the specific instance of VerticalCoord according to VerticalCoordType enum

    :param file_df: Dataframe containing fields of only one file (single path)
    :type file_df: pd.DataFrame
    :param meta_df: Dataframe of all the vertical coordinate meta data for a single grid in a file
    :type meta_df: pd.DataFrame
    :param df: Dataframe all the fields of a single grid and single date of validity from a single file
    :type df: pd.DataFrame
    :raises NotImplementedError: raised when unknown vertical coordinate type is specified
    :return: specific VerticalCoord instance
    :rtype: VerticalCoord
    """
    from .dataframe import add_columns
    if 'vctype' not in df.columns:
        df = add_columns(df, 'ip_info')
    coord_verticale = df.vctype.unique()[0]
    if coord_verticale not in vertical_coord.keys():
        raise NotImplementedError("This type of vertical coordinate doesn't exist")
    try:
        vcoord_inst = vertical_coord[coord_verticale](file_df, meta_df, df)
    except VerticalCoordError as err:
        logging.error(f"Cannot create vertical coordinate {err} - Setting to VerticalCoordUnknown")
        vcoord_inst = VerticalCoordUnknown(file_df, meta_df, df)
    return vcoord_inst

def set_vertical_coordinate_type(df: pd.DataFrame) -> pd.DataFrame:
    """Function that tries to determine the vertical coordinate of the fields

    :param df: input dataframe
    :type df: pd.DataFrame
    :return: output dataframe
    :rtype: pd.DataFrame
    """
    from .dataframe import add_ip_info_columns, get_meta_fields_exists
    from . import VCTYPES

    if 'vctype' not in df.columns:
        df = add_ip_info_columns(df)

        meta_df = df.loc[df.nomvar.isin(["^^", ">>", "^>", "!!", "!!SF", "HY", "P0", "PT"])].reset_index(drop=True)

        no_meta_df = df.loc[~df.nomvar.isin(["^^", ">>", "^>", "!!", "!!SF", "HY", "P0", "PT"])].reset_index(drop=True)

        df_list = []

        groups = no_meta_df.groupby(['grid','ip1_kind'])

        for (grid,ip1_kind), group_df in groups:
            # print('set_vertical_coordinate_type ip1_kind\n',ip1_kind)
            toctoc, p0, e1, pt, hy, sf, vcode = get_meta_fields_exists(meta_df.loc[meta_df.grid == grid])
            # print('set_vertical_coordinate_type get_meta_fields_exists\n',toctoc, p0, e1, pt, hy, sf, vcode)
            # print('set_vertical_coordinate_type group_df\n',group_df.drop(columns='d'))
            if len(vcode) > 1:
                try:
                    index = [divmod(vc, 1000)[0] for vc in vcode].index(ip1_kind)
                    # print('set_vertical_coordinate_type index\n',index)
                except:
                    index = -1

                if index != -1:
                    this_vcode = vcode[index]
                else:
                    this_vcode = -1
            else:
                 this_vcode = vcode[0]

            # print(this_vcode)
            vctyte_df = VCTYPES.loc[(VCTYPES.ip1_kind == ip1_kind) & (VCTYPES.toctoc == toctoc) & (VCTYPES.P0 == p0) & (
                        VCTYPES.E1 == e1) & (VCTYPES.PT == pt) & (VCTYPES.HY == hy) & (VCTYPES.SF == sf) & (VCTYPES.vcode == this_vcode)]

            # print('set_vertical_coordinate_type vctyte_df\n',vctyte_df)
            if not vctyte_df.empty:
                if len(vctyte_df.index) > 1:
                    logging.warning('set_vertical_coordinate_type - more than one match!!!')
                group_df['vctype'] = vctype_dict[vctyte_df.iloc[0]['vctype']]
            else:
                group_df['vctype'] = vctype_dict['UNKNOWN']

            df_list.append(group_df)

        meta_df["vctype"] = vctype_dict['UNKNOWN']

        df_list.append(meta_df)    

        res_df = pd.concat(df_list, ignore_index=True)

        return res_df
    # vctype in columns
    else:
        if df.loc[df.vctype.isna()].empty:
            return df
        else:
            missing_vctype_df = df.loc[df.vctype.isna()]
            correct_vctype_df = df.loc[~df.vctype.isna()]
            missing_vctype_df = missing_vctype_df.drop(columns='vctype')
            new_df = set_vertical_coordinate_type(missing_vctype_df)
            res_df = pd.concat([new_df,correct_vctype_df], ignore_index=True)
            return res_df

    
    # res_df.loc[res_df.nomvar.isin([">>", "^^", "!!", "P0", "PT", "HY", "!!SF"]), "vctype"] = vctype_dict['UNKNOWN']

    # return res_df
    #     ip1_kind_groups = grid.groupby('ip1_kind')
    #     for _, ip1_kind_group in ip1_kind_groups:
    #         # these ip1_kinds are not defined
    #         without_meta = ip1_kind_group.loc[(~ip1_kind_group.ip1_kind.isin(
    #             [-1, 3, 6])) & (~ip1_kind_group.nomvar.isin(["!!", "HY", "P0", "PT", ">>", "^^"]))]
    #         if not without_meta.empty:
    #             ip1_kind = without_meta.iloc[0]['ip1_kind']
    #             # print(vcode)
    #             if len(vcode) > 1:
    #                 for vc in vcode:
    #                     d, _ = divmod(vc, 1000)
    #                     if ip1_kind == d:
    #                         this_vcode = vc
    #                         continue

    #             ip1_kind_group['vctype'] = vctype_dict['UNKNOWN']
    #             #vctype_dict = {'ip1_kind':ip1_kind,'toctoc':toctoc,'P0':p0,'E1':e1,'PT':pt,'HY':hy,'SF':sf,'vcode':vcode}
    #             # print(VCTYPES)
    #             # print(VCTYPES.query('(ip1_kind==%d) and (toctoc==%s) and (P0==%s) and (E1==%s) and (PT==%s) and (HY==%s) and (SF==%s) and (vcode==%d)'%(5,False,True,False,False,False,False,-1)))
    #             # print('\n(ip1_kind==%d) and (toctoc==%s) and (P0==%s) and (E1==%s) and (PT==%s) and (HY==%s) and (SF==%s) and (vcode==%d)'%(ip1_kind,toctoc,p0,e1,pt,hy,sf,this_vcode))
    #             # vctyte_df = VCTYPES.query('(ip1_kind==%d) and (toctoc==%s) and (P0==%s) and (E1==%s) and (PT==%s) and (HY==%s) and (SF==%s) and (vcode==%d)'%(ip1_kind,toctoc,p0,e1,pt,hy,sf,this_vcode))
    #             vctyte_df = VCTYPES.loc[(VCTYPES.ip1_kind == ip1_kind) & (VCTYPES.toctoc == toctoc) & (VCTYPES.P0 == p0) & (
    #                 VCTYPES.E1 == e1) & (VCTYPES.PT == pt) & (VCTYPES.HY == hy) & (VCTYPES.SF == sf) & (VCTYPES.vcode == this_vcode)]
    #             # print(vctyte_df)
    #             if not vctyte_df.empty:
    #                 if len(vctyte_df.index) > 1:
    #                     logging.warning('set_vertical_coordinate_type - more than one match!!!')
    #                 ip1_kind_group['vctype'] = vctype_dict[vctyte_df.iloc[0]['vctype']]
    #         df_list.append(ip1_kind_group)

    # res_df = pd.concat(df_list, ignore_index=True)

    # res_df.loc[res_df.nomvar.isin([">>", "^^", "!!", "P0", "PT", "HY", "!!SF"]), "vctype"] = vctype_dict['UNKNOWN']

    # return res_df

# def set_vertical_coordinate_type2(df: pd.DataFrame) -> pd.DataFrame:
#     """Function that tries to determine the vertical coordinate of the fields

#     :param df: input dataframe
#     :type df: pd.DataFrame
#     :return: output dataframe
#     :rtype: pd.DataFrame
#     """
#     from .dataframe import add_ip_info_columns, get_meta_fields_exists
#     from . import VCTYPES

#     # if 'vctype' in df.columns:
#     #     return df
#     # if 'level' not in df.columns:
#     new_df = add_ip_info_columns(df)
#     # else:
#     #     new_df = copy.deepcopy(df)
#     newdfs = []
#     new_df['vctype'] = vctype_dict['UNKNOWN']
#     grid_groups = new_df.groupby('grid')

#     for _, grid in grid_groups:
#         toctoc, p0, e1, pt, hy, sf, vcode = get_meta_fields_exists(grid)
#         this_vcode = vcode[0]
#         ip1_kind_groups = grid.groupby('ip1_kind')
#         for _, ip1_kind_group in ip1_kind_groups:
#             # these ip1_kinds are not defined
#             without_meta = ip1_kind_group.loc[(~ip1_kind_group.ip1_kind.isin(
#                 [-1, 3, 6])) & (~ip1_kind_group.nomvar.isin(["!!", "HY", "P0", "PT", ">>", "^^"]))]
#             if not without_meta.empty:
#                 ip1_kind = without_meta.iloc[0]['ip1_kind']
#                 # print(vcode)
#                 if len(vcode) > 1:
#                     for vc in vcode:
#                         d, _ = divmod(vc, 1000)
#                         if ip1_kind == d:
#                             this_vcode = vc
#                             continue

#                 ip1_kind_group['vctype'] = vctype_dict['UNKNOWN']
#                 #vctype_dict = {'ip1_kind':ip1_kind,'toctoc':toctoc,'P0':p0,'E1':e1,'PT':pt,'HY':hy,'SF':sf,'vcode':vcode}
#                 # print(VCTYPES)
#                 # print(VCTYPES.query('(ip1_kind==%d) and (toctoc==%s) and (P0==%s) and (E1==%s) and (PT==%s) and (HY==%s) and (SF==%s) and (vcode==%d)'%(5,False,True,False,False,False,False,-1)))
#                 # print('\n(ip1_kind==%d) and (toctoc==%s) and (P0==%s) and (E1==%s) and (PT==%s) and (HY==%s) and (SF==%s) and (vcode==%d)'%(ip1_kind,toctoc,p0,e1,pt,hy,sf,this_vcode))
#                 # vctyte_df = VCTYPES.query('(ip1_kind==%d) and (toctoc==%s) and (P0==%s) and (E1==%s) and (PT==%s) and (HY==%s) and (SF==%s) and (vcode==%d)'%(ip1_kind,toctoc,p0,e1,pt,hy,sf,this_vcode))
#                 vctyte_df = VCTYPES.loc[(VCTYPES.ip1_kind == ip1_kind) & (VCTYPES.toctoc == toctoc) & (VCTYPES.P0 == p0) & (
#                     VCTYPES.E1 == e1) & (VCTYPES.PT == pt) & (VCTYPES.HY == hy) & (VCTYPES.SF == sf) & (VCTYPES.vcode == this_vcode)]
#                 # print(vctyte_df)
#                 if not vctyte_df.empty:
#                     if len(vctyte_df.index) > 1:
#                         logging.warning('set_vertical_coordinate_type - more than one match!!!')
#                     ip1_kind_group['vctype'] = vctype_dict[vctyte_df.iloc[0]['vctype']]
#             newdfs.append(ip1_kind_group)

#     res_df = pd.concat(newdfs, ignore_index=True)

#     res_df.loc[res_df.nomvar.isin([">>", "^^", "!!", "P0", "PT", "HY", "!!SF"]), "vctype"] = vctype_dict['UNKNOWN']

#     return res_df


def get_df_from_vgrid(vgrid_descriptor: vgd.VGridDescriptor, ip1: int, ip2: int) -> pd.DataFrame:
    """Creates a dataframe of one row with the !! field from the rpnpy vgrid_descriptor

    :param vgrid_descriptor: rpnpy vgrid descriptor
    :type vgrid_descriptor: rmn.VGridDescriptor
    :param ip1: ip1 for association to grid
    :type ip1: int
    :param ip2: ip2 for association to grid
    :type ip2: int
    :return: dataframe of one row with the !! field from the vgrid_descriptor
    :rtype: pd.DataFrame
    """
    def create_ig1_for_toctoc(vcoord):
        vers = str(vcoord['VERSION']).zfill(3)
        ig1 = int(''.join([str(vcoord['KIND']), vers]))
        return ig1

    vcoord = vertical_coord_to_dict(vgrid_descriptor)
    ig1 = create_ig1_for_toctoc(vcoord)
    data = vcoord['VTBL']
    meta_df = pd.DataFrame(
        [
            {'nomvar': '!!', 'typvar': 'X', 'etiket': '', 'ni': data.shape[0], 'nj':data.shape[1], 'nk':1, 'dateo':0, 
            'ip1':ip1, 'ip2':ip2, 'ip3':0, 'deet':0, 'npas':0, 'datyp':5, 'nbits':64, 'grtyp':'X', 'ig1':ig1, 'ig2':0, 
            'ig3':0, 'ig4':0, 'datev':0, 'd':data}
        ]
        )
    return meta_df


def vertical_coord_to_dict(vgrid_descriptor: vgd.VGridDescriptor) -> dict:
    """Creates a dictionnary from the rpnpy vgrid_descriptor

    :param vgrid_descriptor: rpnpy vgrid descriptor
    :type vgrid_descriptor: rmn.VGridDescriptor
    :return: dictionnary with 'KIND', 'VERSION' and 'VTBL' values extracted from the vgrid_descriptor
    :rtype: dict
    """
    vcoord = {}
    vcoord['KIND'] = vgd.vgd_get(vgrid_descriptor, 'KIND')
    vcoord['VERSION'] = vgd.vgd_get(vgrid_descriptor, 'VERSION')
    vcoord['VTBL'] = np.asfortranarray(np.squeeze(vgd.vgd_get(vgrid_descriptor, 'VTBL')))
    return vcoord
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
