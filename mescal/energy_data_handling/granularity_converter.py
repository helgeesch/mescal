from enum import Enum

import pandas as pd

from mescal.enums import QuantityTypeEnum
from mescal.energy_data_handling.granularity_analyzer import TimeSeriesGranularityAnalyzer


class GranularityConversionError(Exception):
    pass


class SamplingMethodEnum(Enum):
    UPSAMPLING = 'upsampling'
    DOWNSAMPLING = 'downsampling'
    KEEP = 'keep'


class TimeSeriesGranularityConverter:
    """Converts time series between different granularities while respecting the nature of the quantity.

    This class handles the conversion of time series data between different granularities
    (e.g., hourly to 15-min or vice versa) while properly accounting for the physical
    nature of the quantity being converted:

    - Intensive quantities (e.g., prices, power levels) are replicated when increasing
      granularity and averaged when decreasing granularity.

    - Extensive quantities (e.g., volumes, welfare) are split when increasing
      granularity and summed when decreasing granularity.

    Features:
        - Automatic granularity detection using TimeGranularityAnalyzer
        - Per-day processing to handle missing periods properly and prevent incorrect autofilling of missing days
        - Support for both intensive and extensive quantities
        - Timezone-aware processing including daylight saving transitions
    """
    def __init__(self):
        self._strict_gran_analyzer = TimeSeriesGranularityAnalyzer(strict_mode=True)
        self._non_strict_gran_analyzer = TimeSeriesGranularityAnalyzer(strict_mode=False)

    def _validate_series_format(self, series: pd.Series) -> None:
        if not isinstance(series.index, pd.DatetimeIndex):
            raise TypeError(f"Series index must be DatetimeIndex, got {type(series.index)}")

    def upsample_through_fillna(
            self,
            data: pd.DataFrame | pd.Series,
            quantity_type: QuantityTypeEnum
    ) -> pd.DataFrame | pd.Series:
        if isinstance(data, pd.Series):
            return self._upsample_series(data, quantity_type)

        tmp = data.copy().sort_index()
        idx = tmp.index.tz_convert('UTC') if tmp.index.tz is not None else tmp.index

        if quantity_type == QuantityTypeEnum.EXTENSIVE:
            segment_patterns = tmp.notna().cumsum()

            # Group columns by their segment pattern
            pattern_to_cols = {}
            for col in tmp.columns:
                pattern = tuple(segment_patterns[col].values)  # Convert to tuple to make it hashable
                pattern_to_cols.setdefault(pattern, []).append(col)

            # Process each group of columns with same pattern
            result_pieces = []
            for pattern, cols in pattern_to_cols.items():
                segments = segment_patterns[cols[0]]  # Take segments from first column (all are same)
                piece = (
                    tmp[cols]
                    .groupby([idx.date, idx.hour, segments])
                    .transform(lambda s: s.ffill() / len(s))
                )
                result_pieces.append(piece)

            return pd.concat(result_pieces, axis=1).loc[data.index].rename_axis(data.columns.names, axis=1)
        else:
            return tmp.groupby([idx.date, idx.hour]).ffill().loc[data.index]

    def _upsample_series(self, series: pd.Series, quantity_type: QuantityTypeEnum) -> pd.Series:
        return self.upsample_through_fillna(
            series.to_frame(),
            quantity_type
        ).iloc[:, 0]

    def convert_to_target_index(
            self,
            series: pd.Series,
            target_index: pd.DatetimeIndex,
            quantity_type: QuantityTypeEnum
    ) -> pd.Series:
        self._validate_series_format(series)
        target_gran_series = self._strict_gran_analyzer.get_granularity_as_series_of_timedeltas(target_index)
        _grouped = target_gran_series.groupby(target_gran_series.index.date)
        if _grouped.nunique().max() > 1:
            raise ValueError(f"Found some dates with multiple granularities within same day. Can't handle that!")
        if _grouped.first().nunique() > 1:
            raise ValueError(f"Found multiple granularities. Can't handle that!")
        target_granularity = pd.Timedelta(target_gran_series.values[0])
        return series.groupby(series.index.date).apply(
            lambda x: self._convert_date_to_target_granularity(x, target_granularity, quantity_type)
        ).droplevel(0).rename_axis(series.index.name).rename(series.name)

    def convert_to_target_granularity(
            self,
            series: pd.Series,
            target_granularity: pd.Timedelta,
            quantity_type: QuantityTypeEnum
    ) -> pd.Series:
        """Converts a series to a target granularity."""
        self._validate_series_format(series)
        return series.groupby(series.index.date).apply(
            lambda x: self._convert_date_to_target_granularity(x, target_granularity, quantity_type)
        ).droplevel(0).rename_axis(series.index.name).rename(series.name)

    def _convert_date_to_target_granularity(
        self,
        series: pd.Series,
        target_granularity: pd.Timedelta,
        quantity_type: QuantityTypeEnum
    ) -> pd.Series:
        if len(set(series.index.date)) > 2:
            raise GranularityConversionError('Not intended to use this method.')
        source_gran = self._non_strict_gran_analyzer.get_granularity_as_series_of_minutes(series.index)
        if len(source_gran.unique()) > 1:
            raise GranularityConversionError('Not applicable in case of changing granularity within a single day.')
        source_gran_minutes = source_gran.values[0]
        target_gran_minutes = target_granularity.total_seconds() / 60

        _allowed_granularities = [1, 5, 15, 30, 60, 24*60]
        if target_gran_minutes not in _allowed_granularities:
            raise GranularityConversionError(f'Target granularity must be one of {_allowed_granularities}!')

        if target_gran_minutes > source_gran_minutes:
            sampling = SamplingMethodEnum.DOWNSAMPLING
        elif target_gran_minutes < source_gran_minutes:
            sampling = SamplingMethodEnum.UPSAMPLING
        else:
            sampling = SamplingMethodEnum.KEEP

        if sampling == SamplingMethodEnum.UPSAMPLING:

            scaling_factor = source_gran_minutes / target_gran_minutes
            if (scaling_factor % 1) != 0:
                raise GranularityConversionError('How?')
            else:
                scaling_factor = int(scaling_factor)

            new_index = pd.date_range(
                start=series.index[0],
                end=series.index[-1],
                freq=f"{target_gran_minutes}min",
                tz=series.index.tz
            )

            if quantity_type == QuantityTypeEnum.INTENSIVE:
                return series.reindex(new_index, method='ffill')
            return series.reindex(new_index, method='ffill') / scaling_factor

        elif sampling == SamplingMethodEnum.DOWNSAMPLING:
            groups = series.groupby(pd.Grouper(freq=f"{target_gran_minutes}min"))
            func = 'sum' if quantity_type == QuantityTypeEnum.EXTENSIVE else 'mean'
            return groups.agg(func)

        return series


if __name__ == '__main__':
    import time
    import numpy as np

    converter = TimeSeriesGranularityConverter()

    tz = 'Europe/Berlin'
    hourly_index = pd.date_range('2024-01-01', '2024-12-31 23:45', freq='h', tz=tz)
    quarter_hourly_index = pd.date_range('2024-01-01', '2024-12-31 23:45', freq='15min', tz=tz)

    for qt in [QuantityTypeEnum.INTENSIVE, QuantityTypeEnum.EXTENSIVE]:
        for idx in [hourly_index, quarter_hourly_index]:
            series = pd.Series(100, index=idx)
            series = series.loc[series.index.difference(series.loc['2024-02'].index)]
            print(f'{qt} series: \n{series}')
            for target in [15, 60]:
                start = time.time()
                ts = converter.convert_to_target_granularity(series, pd.Timedelta(minutes=target), qt)
                print(f'To granularity {target}min took {time.time()-start:.2f} seconds:\n{ts}')
            for target_idx in [hourly_index, quarter_hourly_index]:
                start = time.time()
                ts = converter.convert_to_target_index(series, target_idx, qt)
                print(f'To target idx {target_idx[:2]} min took {time.time()-start:.2f} seconds:\n{ts}')

    # Test fillna upsampling
    _values = [100, np.nan, np.nan, np.nan, 200, np.nan, 300, np.nan, 50, np.nan, np.nan, np.nan, np.nan, np.nan]
    mixed_series = pd.Series(_values, index=quarter_hourly_index[:len(_values)])

    for qt in [QuantityTypeEnum.INTENSIVE, QuantityTypeEnum.EXTENSIVE]:
        ts = converter.upsample_through_fillna(mixed_series, qt)
        print(f'Upsampled as {qt}:\n{ts}')