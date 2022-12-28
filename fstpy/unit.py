# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from fstpy import get_unit_by_name
from fstpy.std_dec import get_unit_and_description

from .dataframe import add_columns, add_unit_and_description_columns


class UnitConversionError(Exception):
    pass


class kelvin_to_celsius:
    def __init__(self, bias):
        self.bias = bias

    def __call__(self, v):
        return v - self.bias


class kelvin_to_fahrenheit:
    def __init__(self, bias,   factor):
        self.bias = bias
        self.factor = factor

    def __call__(self, v):
        return (v - 273.15) * 9/5 + 32
        # return v / self.factor - self.bias


class kelvin_to_rankine:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, v):
        return v / self.factor


class celsius_to_kelvin:
    def __init__(self,  bias):
        self.bias = bias

    def __call__(self, v):
        return v + self.bias


class celsius_to_fahrenheit:
    def __init__(self, cbias,   fbias,   ffactor):
        self.cbias = cbias
        self.fbias = fbias
        self.ffactor = ffactor

    def __call__(self, v):
        a = kelvin_to_fahrenheit(self.fbias, self.ffactor)
        b = celsius_to_kelvin(self.cbias)
        return (v * 9/5) + 32
        # return a(b(v))


class celsius_to_rankine:
    def __init__(self, cbias,  rfactor):
        self.cbias = cbias
        self.rfactor = rfactor

    def __call__(self, v):
        a = kelvin_to_rankine(self.rfactor)
        b = celsius_to_kelvin(self.cbias)
        return a(b(v))


class fahrenheit_to_kelvin:
    def __init__(self, bias,   factor):
        self.bias = bias
        self.factor = factor

    def __call__(self, v):
        return (v - 32) * 5/9 + 273.15
        # return (v +  self.bias) * self.factor


class fahrenheit_to_celsius:
    def __init__(self, fbias,   ffactor,   cbias):
        self.fbias = fbias
        self.ffactor = ffactor
        self.cbias = cbias

    def __call__(self, v):
        a = kelvin_to_celsius(self.cbias)
        b = fahrenheit_to_kelvin(self.fbias, self.ffactor)
        return (v - 32) * 5/9
        # return a(b(v))


class fahrenheit_to_rankine:
    def __init__(self,  bias, factor):
        self.bias = bias
        self.factor = factor

    def __call__(self, v):
        a = kelvin_to_rankine(self.factor)
        b = fahrenheit_to_kelvin(self.bias, self.factor)
        return a(b(v))


class rankine_to_kelvin:
    def __init__(self,  factor):
        self.factor = factor

    def __call__(self, v):
        return v * self.factor


class rankine_to_celsius:
    def __init__(self,  rfactor,   cbias):
        self.rfactor = rfactor
        self.cbias = cbias

    def __call__(self, v):
        a = kelvin_to_celsius(self.cbias)
        b = rankine_to_kelvin(self.rfactor)
        return a(b(v))


class rankine_to_fahrenheit:
    def __init__(self, rfactor, fbias,   ffactor):
        self.rfactor = rfactor
        self.fbias = fbias
        self.ffactor = ffactor

    def __call__(self, v):
        a = kelvin_to_fahrenheit(self.fbias, self.ffactor)
        b = rankine_to_kelvin(self.rfactor)
        return a(b(v))


class factor_conversion:
    def __init__(self, from_factor, to_factor):
        self.from_factor = from_factor
        self.to_factor = to_factor

    def __call__(self, v):
        return v * (self.from_factor / self.to_factor)


def get_temperature_converter(unit_from, unit_to):
    from_factor = unit_from.iloc[0]['factor']
    to_factor = unit_to.iloc[0]['factor']
    from_bias = unit_from.iloc[0]['bias']
    to_bias = unit_to.iloc[0]['bias']
    from_name = unit_from.iloc[0]['name']
    to_name = unit_to.iloc[0]['name']
    if (from_name == "kelvin") and (to_name == "celsius"):
        converter = kelvin_to_celsius(to_bias)
        return converter
    if (from_name == "kelvin") and (to_name == "fahrenheit"):
        converter = kelvin_to_fahrenheit(to_bias, to_factor)
        return converter
    if (from_name == "kelvin") and (to_name == "rankine"):
        converter = kelvin_to_rankine(to_factor)
        return converter
    if (from_name == "celsius") and (to_name == "kelvin"):
        converter = celsius_to_kelvin(from_bias)
        return converter
    if (from_name == "celsius") and (to_name == "fahrenheit"):
        converter = celsius_to_fahrenheit(from_bias, to_bias, to_factor)
        return converter
    if (from_name == "celsius") and (to_name == "rankine"):
        converter = celsius_to_rankine(from_bias, to_factor)
        return converter
    if (from_name == "fahrenheit") and (to_name == "kelvin"):
        converter = fahrenheit_to_kelvin(from_bias, from_factor)
        return converter
    if (from_name == "fahrenheit") and (to_name == "celsius"):
        converter = fahrenheit_to_celsius(from_bias, from_factor, to_bias)
        return converter
    if (from_name == "fahrenheit") and (to_name == "rankine"):
        converter = fahrenheit_to_rankine(from_bias, from_factor)
        return converter
    if (from_name == "rankine") and (to_name == "kelvin"):
        converter = rankine_to_kelvin(from_factor)
        return converter
    if (from_name == "rankine") and (to_name == "celsius"):
        converter = rankine_to_celsius(from_factor, to_bias)
        return converter
    if (from_name == "rankine") and (to_name == "fahrenheit"):
        converter = rankine_to_fahrenheit(from_factor, to_bias, to_factor)
        return converter
    return None


def get_converter(unit_from: str, unit_to: str, std: bool = False):
    """Based on unit names contained in fstpy.UNITS database (dataframe),
    attemps to provide the appropriate unit conversion function
    based on unit name and family. The returned function takes a value
    and returns a value value_to = f(value_from).

    :param unit_from: unit name to convert from
    :type unit_from: str
    :param unit_to: unit name to convert to
    :type unit_to: str
    :raises UnitConversionError: Exception
    :return: returns the unit conversion function
    :rtype: function
    """
    from_expression = unit_from.iloc[0]['expression']
    to_expression = unit_to.iloc[0]['expression']
    from_factor = unit_from.iloc[0]['factor']
    to_factor = unit_to.iloc[0]['factor']

    if (unit_from.iloc[0]['name'] == unit_to.iloc[0]['name']):
        return None

    if (from_expression != to_expression) and not std:
        raise UnitConversionError('different unit family')

    if (from_expression != to_expression) and std:
        return None

    if from_expression == 'K':
        converter = get_temperature_converter(unit_from, unit_to)
        # return np.vectorize(converter)
    else:
        converter = factor_conversion(from_factor, to_factor)

    return converter


def unit_convert_array(arr, from_unit_name, to_unit_name='scalar') -> np.ndarray:
    """Converts the data to the specified unit provided in the to_unit_name parameter.

    :param arr: array to be converted
    :type df: np.ndarray
    :param from_unit_name: unit name to convert from
    :type from_unit_name: str
    :param to_unit_name: unit name to convert to, defaults to 'scalar'
    :type to_unit_name: str, optional
    :return: an array containing the converted data
    :rtype: np.ndarray
    """
    unit_to = get_unit_by_name(to_unit_name)
    res_arr = np.copy(arr)
    if (from_unit_name == to_unit_name) or (from_unit_name == 'scalar') or (to_unit_name == 'scalar'):
        return arr
    else:
        unit_from = get_unit_by_name(from_unit_name)
        converter = get_converter(unit_from, unit_to)

        if not(converter is None):
            converted_arr = converter(res_arr)
        else:
            converted_arr = arr

    return converted_arr


def unit_convert(df: pd.DataFrame, to_unit_name='scalar', standard_unit=False) -> pd.DataFrame:
    """Converts the data portion 'd' of all the records of a dataframe to the specified unit
    provided in the to_unit_name parameter. If the standard_unit flag is True, the to_unit_name
    will be ignored and the unit will be based on the standard file variable dictionnary unit
    value instead. This ensures that if a unit conversion was done, the varaible will return
    to the proper standard file unit value. ex. : TT should be in celsius. o.dict can be consulted
    to get the appropriate unit values.

    :param df: dataframe containing records to be converted
    :type df: pd.DataFrame
    :param to_unit_name: unit name to convert to, defaults to 'scalar'
    :type to_unit_name: str, optional
    :param standard_unit: flag to indicate the use of dictionnary units, defaults to False
    :type standard_unit: bool, optional
    :return: a dataframe containing the converted data
    :rtype: pd.DataFrame
    """
    if 'unit' not in df.columns:
        df = add_unit_and_description_columns(df)

    meta_df = df.loc[df.nomvar.isin(["^^", ">>", "^>", "!!", "!!SF", "HY", "P0", "PT"])].reset_index(drop=True)
    # remove meta data from DataFrame
    df = df.loc[~df.nomvar.isin(["^^", ">>", "^>", "!!", "!!SF", "HY", "P0", "PT"])].reset_index(drop=True)
    
    res_df = df.copy(deep=True)
    unit_to = get_unit_by_name(to_unit_name)

    for row in res_df.itertuples():
        current_unit = row.unit
        if (current_unit == to_unit_name):
            continue
        elif (not standard_unit) and ((current_unit == 'scalar') or (to_unit_name == 'scalar')):
            continue
        else:
            unit_from = get_unit_by_name(current_unit)
            if standard_unit:
                to_unit_name, _ = get_unit_and_description(row.nomvar)
                unit_to = get_unit_by_name(to_unit_name)
                converter = get_converter(unit_from, unit_to, True)
            else:
                converter = get_converter(unit_from, unit_to)

            if not(converter is None):
                converted_arr = converter(row.d)
                res_df.at[row.Index, 'd'] = converted_arr
                res_df.at[row.Index, 'unit'] = to_unit_name
                res_df.at[row.Index, 'unit_converted'] = True

    res_df = pd.concat([res_df,meta_df],ignore_index=True)
    if not standard_unit:
        if 'level' not in res_df.columns:
            res_df = add_columns(res_df, columns=['ip_info'])

        res_df = res_df.sort_values(by='level', ascending=res_df.ascending.unique()[0]).reset_index(drop=True)

    return res_df
