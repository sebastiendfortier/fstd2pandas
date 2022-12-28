# -*- coding: utf-8 -*-
import datetime
from typing import Final
from .std_io import decode_ip123

import numpy as np
import rpnpy.librmn.all as rmn
from rpnpy.rpndate import RPNDate

from fstpy import DATYP_DICT, STDVAR
from fstpy.utils import vectorize


class Interval:
    def __init__(self, ip, low, high, kind) -> None:
        self.ip = ip
        self.low = low
        self.high = high
        self.kind = kind
        self.pkind = '' if kind in [-1, 3, 15, 17, 100] else rmn.kindToString(kind).strip()
        pass

    def delta(self):
        if self.kind not in [0, 2, 4, 21, 10]:
            return None
        return self.high-self.low

    def __str__(self):
        return f'{self.ip}:{self.low}{self.pkind}@{self.high}{self.pkind}'
    
    def __eq__(self, other):
        if other is None:
            return False
        return (
            self.ip == other.ip and
            self.low == other.low and
            self.high == other.high and
            self.kind == other.kind
            )

    def __ne__(self, other):
        return not (self == other)

def get_interval(ip1: int, ip2: int, ip3: int, i1: dict, i2: dict, i3: dict) -> 'Interval|None':
    """Gets interval if exists from ip values

    :param ip1: ip1 value
    :type ip1: int
    :param ip2: ip2 value
    :type ip2: int
    :param ip3: ip3 value
    :type ip3: int
    :param i1: decoded ip1 values
    :type i1: dict
    :param i2: decoded ip2 values
    :type i2: dict
    :param i3: decoded ip2 values
    :type i3: dict
    :return: Interval
    :rtype: Interval
    """
    if ip3 >= 32768:
        if (ip1 >= 32768) and (i1['kind'] == i3['kind']):
            return Interval('ip1', i1['v1'], i1['v2'], i1['kind'])
        elif (ip2 >= 32768) and (i2['kind'] == i3['kind']):
            return Interval('ip2', i2['v1'], i2['v2'], i2['kind'])
        else:
            return None
    return None

def get_level_sort_order(kind: int) -> bool:
    """returns the level sort order

    :param kind: level kind
    :type kind: int
    :return: True if the level type is ascending or False otherwise
    :rtype: bool
    """
    # order = {0:'ascending',1:'descending',2:'descending',4:'ascending',5:'descending',21:'ascending'}
    order = {0: True, 3: True, 4: True, 21: True, 100: True,
             1: False, 2: False, 5: False, 6: False, 7: False}
    if kind in order.keys():
        return order[kind]

    return False



def get_forecast_hour(deet: int, npas: int) -> datetime.timedelta:
    """creates a timedelta object in seconds from deet * npas

    :param deet: This is the length of a time step used during a model integration, in seconds.
    :type deet: int
    :param npas: This is the time step number at which the field was written during an integration. The number of the initial time step is 0.
    :type npas: int
    :return: time delta in seconds
    :rtype: datetime.timedelta
    """
    if (deet != 0) or (npas != 0):
        return datetime.timedelta(seconds=int(npas * deet))
    return datetime.timedelta(0)

VCREATE_FORECAST_HOUR: Final = vectorize(get_forecast_hour, otypes=['timedelta64[ns]'])  # ,otypes=['timedelta64[ns]']

def get_data_type_str(datyp: int):
    """gets the data type string from the datyp int

    :param datyp: data type int value
    :type datyp: int
    :return: string eqivalent of the datyp int value
    :rtype: str
    """
    return DATYP_DICT[datyp]

VCREATE_DATA_TYPE_STR: Final = vectorize(get_data_type_str, otypes=['str'])


def get_ip_info(nomvar:str, ip1: int, ip2: int, ip3: int):
    """gets all relevant level info from the ip1 int value

    :param ip1: encoded value stored in ip1
    :type ip1: int
    :return: level value, kind and kind str obtained from decoding ip1 and bools representing if the level is a surface level, if it follows topography and its sort order.
    :rtype: float,int,str,bool,bool,bool
    """
    # iii1, iii2, iii3 = rmn.DecodeIp(ip1,ip2,ip3)
    i1, i2, i3 = decode_ip123(nomvar, ip1, ip2, ip3) 
    # if nomvar not in ['>>','^^','!!','^>']:
    #     print(nomvar,iii1,i1,iii2,i2,iii3,i3)

    #     print(nomvar ,[(iii1.v1,i1['v1']) if (iii1.v1 != i1['v1']) else True, (iii2.v1,i2['v1']) if (iii2.v1 != i2['v1']) else True, (iii3.v1,i3['v1']) if (iii3.v1 != i3['v1']) else True])

    surface = is_surface(i1['kind'], i1['v1'])

    follow_topography = level_type_follows_topography(i1['kind'])

    ascending = get_level_sort_order(i1['kind'])

    interval = get_interval(ip1, ip2, ip3, i1, i2, i3)

    return i1['v1'], i1['kind'], i1['kinds'], i2['v1'], i2['kind'], i2['kinds'], i3['v1'], i3['kind'], i3['kinds'], surface, follow_topography, ascending, interval

VCREATE_IP_INFO: Final = vectorize(get_ip_info, otypes=['float32', 'int32', 'str', 'float32', 'int32', 'str', 'float32', 'int32', 'str', 'bool', 'bool', 'bool', 'object'])


def get_unit_and_description(nomvar):
    """Reads the Standard file dictionnary and gets the unit and description associated with the variable name

    :param nomvar: name of the variable
    :type nomvar: str
    :return: unit name and description
    :rtype: str,str

    >>> get_unit_and_description('TT')
    'Air Temperature' 'celsius'
    """
    unit = STDVAR.loc[STDVAR['nomvar'] == f'{nomvar}']['unit'].values
    description = STDVAR.loc[STDVAR['nomvar'] == f'{nomvar}']['description_en'].values
    if len(description):
        description = description[0]
    else:
        description = ''
    if len(unit):
        unit = unit[0]
    else:
        unit = 'scalar'
    return unit, description

VGET_UNIT_AND_DESCRIPTION: Final = vectorize(get_unit_and_description, otypes=['str', 'str'])

# written by Micheal Neish creator of fstd2nc
def convert_rmndate_to_datetime(date: int) -> 'datetime.datetime|None':
    """returns a datetime object of the decoded RMNDate int

    :param date: RMNDate int value
    :type date: int
    :return: datetime object of the decoded date
    :rtype: datetime.datetime

    >>> convert_rmndate_to_datetime(442998800)
    datetime.datetime(2020, 7, 14, 12, 0)
    """
    dummy_stamps = (0, 10101011)
    if date not in dummy_stamps:
        return RPNDate(int(date)).toDateTime().replace(tzinfo=None)
    else:
        return None

VCONVERT_RMNDATE_TO_DATETIME: Final = vectorize(convert_rmndate_to_datetime, otypes=['datetime64'])  # ,otypes=['datetime64']

def is_surface(ip1_kind: int, level: float) -> bool:
    """Return a bool that tell us if the level is a surface level

    :param ip1_kind: kind of level
    :type ip1_kind: int
    :param level: value of the level
    :type level: float
    :return: True if the level is a surface level else False
    :rtype: bool

    >>> is_surface(5,0.36116)
    False
    """
    meter_levels = np.arange(0., 10.5, .5).tolist()
    if (ip1_kind == 5) and (level == 1):
        return True
    elif (ip1_kind == 4) and (level in meter_levels):
        return True
    elif (ip1_kind == 1) and (level == 1):
        return True
    else:
        return False


def level_type_follows_topography(ip1_kind: int) -> bool:
    """Returns True if the kind of level is a kind that follows topography

    :param ip1_kind: level type
    :type ip1_kind: int
    :return: True if the kind of level is a kind that follows topography else False
    :rtype: bool

    >>> level_type_follows_topography(5)
    True
    """
    if ip1_kind == 1:
        return True
    elif ip1_kind == 4:
        return True
    elif ip1_kind == 5:
        return True
    else:
        return False


def get_grid_identifier(nomvar: str, ip1: int, ip2: int, ig1: int, ig2: int) -> str:
    """Create a grid identifer from ip2,ip2 or ig1,ig2 depending of the varibale.
    Meta information like >> have their grid identifiers strored in ip1,and ip2,
    while regular viables have them strored in ig1 and ig2

    :param nomvar: name of the variable
    :type nomvar: str
    :param ip1: ip1 value
    :type ip1: int
    :param ip2: ip2 value
    :type ip2: int
    :param ig1: ig1 value
    :type ig1: int
    :param ig2: ig2 value
    :type ig2: int
    :return: concatenation of ig1,ig2 or ip1,ip2 depending on variable name
    :rtype: str

    >>> get_grid_identifier('TT',94733000,6,33792,77761)
    '3379277761'
    """
    nomvar = nomvar.strip()
    if nomvar in ["^>", ">>", "^^", "!!", "!!SF"]:
        grid = "".join([str(ip1), str(ip2)])
    elif nomvar == "HY":
        grid = 'None'
    else:
        grid = "".join([str(ig1), str(ig2)])
    return grid

VCREATE_GRID_IDENTIFIER: Final = vectorize(get_grid_identifier, otypes=['str'])

def get_parsed_etiket(raw_etiket: str):
    """parses the etiket of a standard file to get label, run, implementation and ensemble member if available

    :param raw_etiket: raw etiket before parsing
    :type raw_etiket: str
    :return: the parsed etiket, run, implementation and ensemble member
    :rtype: str

    >>> get_parsed_etiket('')
    ('', '', '', '')
    >>> get_parsed_etiket('R1_V710_N')
    ('_V710_', 'R1', 'N', '')
    """
    import re
    label = raw_etiket
    run = None
    implementation = None
    ensemble_member = None

    match_run = "[RGPEAIMWNC_][\\dRLHMEA_]"
    match_main_cmc = "\\w{5}"
    match_main_spooki = "\\w{6}"
    match_implementation = "[NPX]"
    match_ensemble_member = "\\w{3}"
    match_end = "$"

    re_match_cmc_no_ensemble = match_run + \
        match_main_cmc + match_implementation + match_end
    re_match_cmc_ensemble = match_run + match_main_cmc + \
        match_implementation + match_ensemble_member + match_end
    re_match_spooki_no_ensemble = match_run + \
        match_main_spooki + match_implementation + match_end
    re_match_spooki_ensemble = match_run + match_main_spooki + \
        match_implementation + match_ensemble_member + match_end

    if re.match(re_match_cmc_no_ensemble, raw_etiket):
        run = raw_etiket[:2]
        label = raw_etiket[2:7]
        implementation = raw_etiket[7]
    elif re.match(re_match_cmc_ensemble, raw_etiket):
        run = raw_etiket[:2]
        label = raw_etiket[2:7]
        implementation = raw_etiket[7]
        ensemble_member = raw_etiket[8:11]
    elif re.match(re_match_spooki_no_ensemble, raw_etiket):
        run = raw_etiket[:2]
        label = raw_etiket[2:8]
        implementation = raw_etiket[8]
    elif re.match(re_match_spooki_ensemble, raw_etiket):
        run = raw_etiket[:2]
        label = raw_etiket[2:8]
        implementation = raw_etiket[8]
        ensemble_member = raw_etiket[9:12]
    else:
        label = raw_etiket
    return label, run, implementation, ensemble_member

VPARSE_ETIKET: Final = vectorize(get_parsed_etiket, otypes=['str', 'str', 'str', 'str'])
