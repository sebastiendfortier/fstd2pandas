# -*- coding: utf-8 -*-
import logging
import pandas as pd
import dask.array.ma as ma
from . import BASE_COLUMNS
import copy

from .dataframe_utils import metadata_cleanup 
# import numpy.ma as ma
from .utils import initializer

class RecoverMaskError(Exception):
    pass


class RecoverMask():
    """Recovers the masks from the data when available. If a field has a mask, then the resulting dataframe will have 
       the mask rows added and the mask removed from the arrays (np.ma.array -> np.array).

    :param df: input dataframe
    :type df: pd.DataFrame
    :param mask_typvar: typvar to identify mask field, defaults to '@@'
    :type mask_typvar: str, optional
    :param keep_value: bool as int that indicates the mask values to keep, defaults to 1
    :type keep_value: int, optional
    """
    @initializer
    def __init__(self, df: pd.DataFrame, mask_typvar: str = '@@', keep_value: int = 1):
        self.validate_typvar()
        self.validate_value()
        self.get_dataframes()

    def validate_value(self):
        if self.keep_value not in [0,1]:
            raise RecoverMaskError(f'keep_value can only be 0 or 1! provided {self.keep_value}')

    def validate_typvar(self):
        if len(self.mask_typvar) != 2:
            raise RecoverMaskError(f'mask_typvar has to be 2 characters in length! provided {self.mask_typvar}')

    def get_dataframes(self):
        """creates self.meta_df and self.no_meta_df"""
        self.meta_df = self.df.loc[self.df.nomvar.isin(
            ["^^", ">>", "^>", "!!", "!!SF", "HY", "P0", "PT"])].reset_index(drop=True)

        self.no_meta_df = self.df.loc[~self.df.nomvar.isin(
            ["^^", ">>", "^>", "!!", "!!SF", "HY", "P0", "PT"])].reset_index(drop=True)

        self.masked_df = self.no_meta_df.loc[self.no_meta_df.typvar.str.contains('@')]
        self.not_masked_df = self.no_meta_df.loc[~self.no_meta_df.typvar.str.contains('@')]

    def compute(self)->pd.DataFrame:
        logging.info('RecoverMask - compute')
        cols = [col for col in BASE_COLUMNS if col not in ['typvar', 'datyp', 'nbits', 'datev', 'd']]
        groups = self.masked_df.groupby(cols)
        df_list = []
        df_list.append(self.meta_df)
        df_list.append(self.not_masked_df)
        for _, var_df in groups:
            if len(var_df.index) == 2:
                df_list.append(var_df)
                continue

            if len(var_df.index) != 1:
                raise RecoverMaskError('There should only be 1 row per group!')

            mask_df = copy.deepcopy(var_df)
            mask_df['typvar'] = self.mask_typvar
            mask_df['datyp'] = 2
            mask_df['nbits'] = 1
            var = var_df.iloc[0].d

            var_df['d'] = [ma.getdata(var)]
            mask = ma.getmaskarray(var)
            # if isinstance(var,)
            if self.keep_value == 1:
                mask = (~mask).astype('uint32')
            else:
                mask = mask.astype('uint32')

            mask_df['d'] = [mask]
            df_list.append(var_df)
            df_list.append(mask_df)



        new_list = []
        for df in df_list:
            if not df.empty:
                new_list.append(df)

        if not len(new_list):
            raise RecoverMaskError('No results were produced')

        # merge all results together
        res_df = pd.concat(new_list, ignore_index=True)

        res_df = metadata_cleanup(res_df)

        return res_df



