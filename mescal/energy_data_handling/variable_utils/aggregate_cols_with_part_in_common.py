import numpy as np
import pandas as pd

from mescal.utils.pandas_utils.set_new_column import set_column


class AggregatedColumnAppender:
    """
    Adds an aggregated column to a pandas DataFrame by summing columns
    that share a common identifier in their names.
    """

    def __init__(
            self,
            in_common_part: str,
            agg_col_name_prefix: str = None,
            agg_col_name_suffix: str = None,
    ):
        self._in_common_part = in_common_part
        self._agg_col_name_prefix = agg_col_name_prefix or ''
        self._agg_col_name_suffix = agg_col_name_suffix or ''

    def add_aggregated_column(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = df.columns.get_level_values(0).unique()
        cols_with_common_part = [x for x in cols if self._in_common_part in x]

        df_in_common = df[cols_with_common_part]
        if df.columns.nlevels == 1:
            dff = df_in_common.sum(axis=1)
            dff.loc[df_in_common.isna().all(axis=1)] = np.nan
        else:
            _groupby = list(range(1, df.columns.nlevels))
            dff = df_in_common.T.groupby(level=_groupby).sum().T
            _all_na = df_in_common.isna().T.groupby(level=_groupby).all().T
            if _all_na.any().any():
                for c in _all_na.columns:
                    dff.loc[_all_na[c], c] = np.nan

        new_col_name = f'{self._agg_col_name_prefix}{self._in_common_part}{self._agg_col_name_suffix}'
        df = set_column(df, new_col_name, dff)
        return df


if __name__ == '__main__':
    data = pd.DataFrame({
        'volume_curve': [100, 200, 300],
        'volume_block': [50, 70, 100],
        'bing_bong': [30, 40, 50],
        'king_kong': [20, 25, 30],
    })

    appender = AggregatedColumnAppender(in_common_part='volume', agg_col_name_suffix='_total')
    result_df = appender.add_aggregated_column(data)

    print(result_df)
