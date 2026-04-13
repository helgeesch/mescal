from enum import Enum

import pandas as pd

from mesqual.enums import QuantityTypeEnum


class GranularityConversionError(Exception):
    """Exception raised when granularity conversion operations fail."""
    pass


class SamplingMethodEnum(Enum):
    UPSAMPLING = 'upsampling'
    DOWNSAMPLING = 'downsampling'
    KEEP = 'keep'


class TimeSeriesGranularityConverter:
    """Converts time series between different granularities while respecting quantity type.

    Handles both upsampling (coarser → finer) and downsampling (finer → coarser):

    - Intensive quantities (prices, power): ffill when upsampling, mean when downsampling
    - Extensive quantities (volumes, welfare): ffill/÷spread when upsampling, sum when downsampling

    Works on both Series and DataFrames, including DataFrames with mixed-granularity
    columns (e.g., some columns hourly, others 15-min within the same DataFrame).
    The conversion is applied per-column — columns already at the target frequency
    pass through unchanged.

    Supports time series with granularity transitions (e.g., hourly before Oct 2025,
    15-min after). The sampling direction is determined per calendar day, and
    contiguous blocks of the same direction are processed together.

    Uses pandas resample directly, avoiding explicit granularity analysis of the source data.
    Forward-fill is bounded per calendar day to prevent bleeding across data gaps.
    """

    def convert(
        self,
        data: pd.DataFrame | pd.Series,
        target_freq: str | pd.Timedelta,
        quantity_type: QuantityTypeEnum,
    ) -> pd.DataFrame | pd.Series:
        """Convert time series data to a target frequency.

        Args:
            data: Input time series with DatetimeIndex.
            target_freq: Target frequency as a Timedelta or timedelta-compatible string
                         (e.g., "15min", "1h", pd.Timedelta(hours=1)).
            quantity_type: INTENSIVE (prices, power) or EXTENSIVE (volumes, welfare).

        Returns:
            Converted data, same type as input (Series or DataFrame).
        """
        is_series = isinstance(data, pd.Series)
        df = data.to_frame() if is_series else data

        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(f"Index must be DatetimeIndex, got {type(df.index)}")

        target_td = target_freq if isinstance(target_freq, pd.Timedelta) else pd.Timedelta(target_freq)

        # Determine sampling direction per day based on each day's median index step
        dates = pd.Series(df.index.date, index=df.index)
        day_gran = df.index.to_series().groupby(dates).transform(lambda g: g.diff().median())

        direction = pd.Series(SamplingMethodEnum.KEEP, index=df.index)
        direction[day_gran > target_td] = SamplingMethodEnum.UPSAMPLING
        direction[day_gran < target_td] = SamplingMethodEnum.DOWNSAMPLING

        # KEEP days with sparse columns (NaN) still need filling
        keep_mask = direction == SamplingMethodEnum.KEEP
        if keep_mask.any() and df[keep_mask].isna().any().any():
            direction[keep_mask] = SamplingMethodEnum.UPSAMPLING

        # Fast path: uniform direction across the entire series
        unique_dirs = direction.unique()
        if len(unique_dirs) == 1:
            result = self._apply(df, unique_dirs[0], target_td, quantity_type)
        else:
            # Process contiguous blocks of the same direction separately
            # so that resample doesn't create spurious entries in the gaps
            block_ids = (direction != direction.shift()).cumsum()
            parts = []
            for _, block_rows in direction.groupby(block_ids):
                block = df.loc[block_rows.index]
                parts.append(self._apply(block, block_rows.iloc[0], target_td, quantity_type))
            result = pd.concat(parts).sort_index()

        if is_series:
            return result.iloc[:, 0].rename(data.name)
        return result

    def convert_to_target_index(
        self,
        data: pd.DataFrame | pd.Series,
        target_index: pd.DatetimeIndex,
        quantity_type: QuantityTypeEnum,
    ) -> pd.DataFrame | pd.Series:
        """Convert data to match a specific target DatetimeIndex.

        Derives the target frequency from the index, converts, then reindexes
        to ensure the output aligns exactly with the requested timestamps.
        """
        target_freq = target_index.to_series().diff().median()
        result = self.convert(data, target_freq, quantity_type)
        return result.reindex(target_index)

    def upsample_through_fillna(
        self,
        data: pd.DataFrame | pd.Series,
        quantity_type: QuantityTypeEnum,
    ) -> pd.DataFrame | pd.Series:
        """Fill sparse columns at the existing index frequency.

        For DataFrames where some columns have coarser data (NaN at intermediate
        timestamps), this fills the gaps respecting the quantity type — replicating
        for intensive quantities, distributing for extensive quantities.
        """
        freq = data.index.to_series().diff().min()
        return self.convert(data, freq, quantity_type)

    # kept for backward compatibility
    def convert_to_target_granularity(
        self,
        data: pd.DataFrame | pd.Series,
        target_granularity: pd.Timedelta,
        quantity_type: QuantityTypeEnum,
    ) -> pd.DataFrame | pd.Series:
        return self.convert(data, target_granularity, quantity_type)

    # ── internals ────────────────────────────────────────────────────────

    def _apply(
        self,
        df: pd.DataFrame,
        direction: SamplingMethodEnum,
        target_td: pd.Timedelta,
        quantity_type: QuantityTypeEnum,
    ) -> pd.DataFrame:
        if direction == SamplingMethodEnum.UPSAMPLING:
            return self._upsample(df, target_td, quantity_type)
        if direction == SamplingMethodEnum.DOWNSAMPLING:
            return self._downsample(df, target_td, quantity_type)
        return df

    @staticmethod
    def _downsample(
        df: pd.DataFrame,
        target_td: pd.Timedelta,
        quantity_type: QuantityTypeEnum,
    ) -> pd.DataFrame:
        resampler = df.resample(target_td)
        if quantity_type == QuantityTypeEnum.EXTENSIVE:
            return resampler.sum(min_count=1)
        return resampler.mean()

    @staticmethod
    def _upsample(
        df: pd.DataFrame,
        target_td: pd.Timedelta,
        quantity_type: QuantityTypeEnum,
    ) -> pd.DataFrame:
        resampled = df.resample(target_td).asfreq()
        days = resampled.index.normalize()

        if quantity_type == QuantityTypeEnum.INTENSIVE:
            return resampled.groupby(days).ffill()

        # Extensive: forward-fill within days, then divide by spread factor.
        # Each non-NaN value starts a new "segment" (via cumsum of the notna mask).
        # The segment size tells us over how many target periods the original value
        # was spread — that's the divisor.  Columns already at the target frequency
        # have segment size 1 everywhere, so they pass through unchanged.
        filled = resampled.groupby(days).ffill()
        segments = resampled.notna().cumsum()
        spread = segments.apply(
            lambda col: col.groupby([days, col]).transform('size')
        )
        return filled / spread


if __name__ == '__main__':
    import time
    import numpy as np

    converter = TimeSeriesGranularityConverter()

    tz = 'Europe/Berlin'
    hourly_index = pd.date_range('2024-01-01', '2024-12-31 23:45', freq='h', tz=tz)
    quarter_hourly_index = pd.date_range('2024-01-01', '2024-12-31 23:45', freq='15min', tz=tz)

    # ── basic Series round-trips ──
    for qt in [QuantityTypeEnum.INTENSIVE, QuantityTypeEnum.EXTENSIVE]:
        for idx in [hourly_index, quarter_hourly_index]:
            series = pd.Series(100, index=idx)
            series = series.loc[series.index.difference(series.loc['2024-02'].index)]
            print(f'{qt} series:\n{series}')
            for target in ['15min', '1h']:
                start = time.time()
                ts = converter.convert(series, target, qt)
                print(f'  → {target} took {time.time()-start:.2f}s:\n{ts}')
            for target_idx in [hourly_index, quarter_hourly_index]:
                start = time.time()
                ts = converter.convert_to_target_index(series, target_idx, qt)
                print(f'  → target idx took {time.time()-start:.2f}s:\n{ts}')

    # ── mixed-granularity DataFrame ──
    print('\n── mixed-granularity DataFrame ──')
    df = pd.DataFrame({
        'hourly_col': pd.Series(100, index=hourly_index),
        'qh_col': pd.Series(10, index=quarter_hourly_index),
    })
    print(f'Mixed DF shape: {df.shape}, NaN count:\n{df.isna().sum()}')
    for qt in [QuantityTypeEnum.INTENSIVE, QuantityTypeEnum.EXTENSIVE]:
        start = time.time()
        result = converter.convert(df, '15min', qt)
        print(f'{qt} → 15min took {time.time()-start:.2f}s, shape={result.shape}')
        print(result.head(8))

    # ── fillna upsampling ──
    print('\n── upsample_through_fillna ──')
    _values = [100, np.nan, np.nan, np.nan, 200, np.nan, 300, np.nan,
               50, np.nan, np.nan, np.nan, np.nan, np.nan]
    mixed_series = pd.Series(_values, index=quarter_hourly_index[:len(_values)])
    for qt in [QuantityTypeEnum.INTENSIVE, QuantityTypeEnum.EXTENSIVE]:
        ts = converter.upsample_through_fillna(mixed_series, qt)
        print(f'Upsampled as {qt}:\n{ts}')

    # ── mixed-granularity time series (hourly Jan-Jun, 15min Jul-Dec) ──
    print('\n── mixed-granularity time series (simulating mid-year resolution change) ──')
    hourly_part = pd.Series(100, index=pd.date_range('2024-01-01', '2024-06-30 23:00', freq='h', tz=tz))
    qh_part = pd.Series(10, index=pd.date_range('2024-07-01', '2024-12-31 23:45', freq='15min', tz=tz))
    mixed_series = pd.concat([hourly_part, qh_part])
    print(f'Mixed series length: {len(mixed_series)}')
    print(f'  hourly part: {len(hourly_part)}, 15min part: {len(qh_part)}')

    for qt in [QuantityTypeEnum.INTENSIVE, QuantityTypeEnum.EXTENSIVE]:
        for target in ['15min', '1h']:
            start = time.time()
            ts = converter.convert(mixed_series, target, qt)
            print(f'  {qt} → {target} took {time.time()-start:.2f}s, len={len(ts)}')
            # Verify: check a few values from each part
            print(f'    Jan sample: {ts.iloc[:4].values}')
            print(f'    Jul sample: {ts.loc["2024-07-01"].iloc[:4].values}')