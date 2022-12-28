# -*- coding: utf-8 -*-
import datetime

import rpnpy.librmn.all as rmn
from rpnpy.rpndate import RPNDate

from fstpy import DATYP_DICT


def create_encoded_etiket(label: str, run: str, implementation: str, ensemble_member: str) -> str:
    """Creates a new etiket based on label, run, implementation and ensemble member attributes

    :param label: label string
    :type label: str
    :param run: model run string
    :type run: str
    :param implementation: implementation string
    :type implementation: str
    :param ensemble_member: ensemble member number as string
    :type ensemble_member: str
    :return: an etiket composed of supplied parameters
    :rtype: str
    """
    etiket =  label

    if run != 'None':
        etiket = run + label
    if implementation != 'None':
        etiket = etiket + implementation
    if ensemble_member != 'None':
        etiket = etiket + ensemble_member

    return etiket


def create_encoded_dateo(date_of_observation: datetime.datetime) -> int:
    """Create a RMNDate int from a datetime object

    :param date_of_observation: date of observation as a datetime object
    :type date_of_observation: datetime.datetime
    :return: dateo as a RMNDate int 
    :rtype: int
    """

    return RPNDate(date_of_observation, dt=0, nstep=0).dateo


def create_encoded_npas_and_ip2(forecast_hour: datetime.timedelta, deet: int) -> tuple:
    """Creates npas and ip2 from the forecast_hour and deet attributes

    :param forecast_hour: forecast hours in seconds
    :type forecast_hour: datetime.timedelta
    :param deet: length of a time step in seconds - usually invariable - relates to model ouput times
    :type deet: int
    :return: new calculated npas and ip2
    :rtype: tuple
    """
    # ip2 = 6, deet = 300, np = 72
    #fhour = 21600
    #npas = hours/deet
    seconds = forecast_hour.total_seconds()
    npas = int(seconds / deet)
    ip2 = seconds / 3600.
    ip2_code = create_encoded_ip2(ip2, rmn.KIND_HOURS)
    return npas, ip2_code


def create_encoded_ip1(level: float, ip1_kind: int,mode:int=rmn.CONVIP_ENCODE) -> int:
    """returns an encoded ip1 from level and kind

    :param level: level value
    :type level: float
    :param ip1_kind: kind value as int
    :type ip1_kind: int
    :return: encoded ip1
    :rtype: int
    """
  
    return rmn.convertIp(mode,level,ip1_kind)


def create_encoded_ip2(level: float, ip2_kind: int) -> int:
    """returns an encoded ip2 from level and kind

    :param level: level value
    :type level: float
    :param ip2_kind: kind value as int
    :type ip2_kind: int
    :return: encoded ip2
    :rtype: int
    """
    rp1 = rmn.FLOAT_IP(0, 0, rmn.KIND_ARBITRARY)
    rp2 = rmn.FLOAT_IP(level, level, ip2_kind)
    return rmn.EncodeIp(rp1, rp2, rp1)[1]


def create_encoded_ips(level: float, ip1_kind: int, ip2_dec: float, ip2_kind: int, ip3_dec: float, ip3_kind: int) -> tuple:
    """Returns encoded ip1,ip2 and ip3 from values and kinds

    :param level: level value
    :type level: float
    :param ip1_kind: ip1 kind value
    :type ip1_kind: int
    :param ip2_dec: decoded ip2 value
    :type ip2_dec: float
    :param ip2_kind: ip2 kind  value
    :type ip2_kind: int
    :param ip3_dec: decoded ip3 valued
    :type ip3_dec: float
    :param ip3_kind: ip3 kind value
    :type ip3_kind: int
    :return: encoded ip1,ip2 and ip3 values
    :rtype: tuple
    """
    ip1 = create_encoded_ip1(level, ip1_kind)
    ip2 = create_encoded_ip1(ip2_dec, ip2_kind)
    ip3 = create_encoded_ip1(ip3_dec, ip3_kind)
    return ip1, ip2, ip3


def create_encoded_datyp(data_type_str: str) -> int:
    """creates an encoded datyp value from a data type string

    :param data_type_str: possible values 'X','R','I','S','E','F','A','Z','i','e','f'
    :type data_type_str: str
    :return: an ecoded datyp value
    :rtype: int
    """
    new_dict = {v: k for k, v in DATYP_DICT.items()}
    return new_dict[data_type_str]


def modifiers_to_typvar2(zapped: bool, filtered: bool, interpolated: bool, unit_converted: bool, bounded: bool, ensemble_extra_info: bool, multiple_modifications: bool) -> str:
    """Creates the second lette of the typvar from the supplied flags"""
    number_of_modifications = 0
    typvar2 = ''
    if zapped == True:
        number_of_modifications += 1
        typvar2 = 'Z'
    if filtered == True:
        number_of_modifications += 1
        typvar2 = 'F'
    if interpolated == True:
        number_of_modifications += 1
        typvar2 = 'I'
    if unit_converted == True:
        number_of_modifications += 1
        typvar2 = 'U'
    if bounded == True:
        number_of_modifications += 1
        typvar2 = 'B'
    if ensemble_extra_info == True:
        number_of_modifications += 1
        typvar2 = '!'
    if multiple_modifications == True:
        number_of_modifications += 1
        typvar2 = 'M'
    if number_of_modifications > 1:
        # more than one modification has been done. Force M
        typvar2 = 'M'
    return typvar2

def encode_ip2_and_ip3_as_time_interval(df):
    for row in df.itertuples():
        if row.nomvar in ['>>', '^^', '^>', '!!', 'P0', 'PT']:
            continue
        ip2 = row.ip2
        ip3 = row.ip3
        rp1a = rmn.FLOAT_IP(0., 0., rmn.LEVEL_KIND_PMB)
        rp2a = rmn.FLOAT_IP( ip2,  ip3, rmn.TIME_KIND_HR)
        rp3a = rmn.FLOAT_IP( ip2-ip3,  0, rmn.TIME_KIND_HR)
        (_, ip2, ip3) = rmn.EncodeIp(rp1a, rp2a, rp3a)
        df.at[row.Index,'ip2'] = ip2
        df.at[row.Index,'ip3'] = ip3
    return df    
