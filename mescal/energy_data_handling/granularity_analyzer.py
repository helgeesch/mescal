import warnings
from functools import wraps
import pandas as pd


class GranularityError(Exception):
    pass


def _validate_dt_index(func):
    @wraps(func)
    def wrapper(self, dt_index, *args, **kwargs):
        if not isinstance(dt_index, pd.DatetimeIndex):
            try:
                dt_index = pd.DatetimeIndex(dt_index)
            except Exception as e:
                raise TypeError(f"Input must be convertible to DatetimeIndex, got {type(dt_index)}") from e
        return func(self, dt_index, *args, **kwargs)

    return wrapper


class TimeSeriesGranularityAnalyzer:
    """Analyzes and validates time granularity in DatetimeIndex sequences.

    This class provides tools for working with time series data that may have varying
    granularities (e.g., hourly, quarter-hourly). It's particularly useful for
    electricity market data analysis where different market products can have
    different time resolutions.

    Features:
        - Granularity detection for time series data
        - Support for mixed granularities within the same series
        - Strict mode for validation scenarios
        - Per-day granularity analysis

    Args:
        strict_mode: If True, raises GranularityError when multiple granularities
            are detected. If False, only issues warnings. Default is True.

    Example:
        >>> analyzer = TimeSeriesGranularityAnalyzer()
        >>> index = pd.date_range('2024-01-01', periods=24, freq='h')
        >>> analyzer.get_granularity_as_timedelta(index)
        Timedelta('1 hours')
    """

    def __init__(self, strict_mode: bool = True):
        self._strict_mode = strict_mode

    @property
    def strict_mode(self) -> bool:
        return self._strict_mode

    @strict_mode.setter
    def strict_mode(self, value: bool):
        self._strict_mode = value

    @_validate_dt_index
    def get_granularity_as_series_of_timedeltas(self, dt_index: pd.DatetimeIndex) -> pd.Series:
        s = pd.Series(index=dt_index)
        s = s.groupby(dt_index.date).apply(lambda g: self._get_granularity_for_day(g.index))
        s = s.droplevel(0, axis=0)
        if dt_index.name is not None:
            s.index.name = dt_index.name
        return s

    @_validate_dt_index
    def get_granularity_as_series_of_minutes(self, dt_index: pd.DatetimeIndex) -> pd.Series:
        return self.get_granularity_as_series_of_timedeltas(dt_index).apply(lambda x: x.total_seconds() / 60)

    @_validate_dt_index
    def get_granularity_as_series_of_hours(self, dt_index: pd.DatetimeIndex) -> pd.Series:
        return self.get_granularity_as_series_of_timedeltas(dt_index).apply(lambda x: x.total_seconds() / 3600)

    def _get_granularity_for_day(self, dt_index: pd.DatetimeIndex) -> pd.Series:
        gran = dt_index.to_series().diff().shift(-1)
        if len(gran) > 1:
            gran.iloc[-1] = gran.iloc[-2]

        if gran.nunique() > 1:
            msg = f'Multiple granularities identified within a day: {gran.unique()}'
            if self._strict_mode:
                raise GranularityError(msg)
            warnings.warn(msg)
        return gran

    @_validate_dt_index
    def get_granularity_as_timedelta(self, dt_index: pd.DatetimeIndex) -> pd.Timedelta:
        if len(dt_index) == 0:
            return pd.Timedelta(0)

        gran = self.get_granularity_as_series_of_timedeltas(dt_index)
        first_gran = gran.iloc[0]

        if gran.nunique() > 1:
            msg = (f'Multiple granularities found: {gran.unique()}. '
                   f'Using {first_gran} as the reference granularity.')
            if self._strict_mode:
                raise GranularityError(msg)
            warnings.warn(msg)
        return first_gran

    @_validate_dt_index
    def get_granularity_as_hours(self, dt_index: pd.DatetimeIndex) -> float:
        return self.get_granularity_as_timedelta(dt_index).total_seconds() / 3600

    @_validate_dt_index
    def get_granularity_as_minutes(self, dt_index: pd.DatetimeIndex) -> float:
        return self.get_granularity_as_timedelta(dt_index).total_seconds() / 60

    @_validate_dt_index
    def validate_constant_granularity(self, dt_index: pd.DatetimeIndex, expected_hours: float) -> bool:
        actual_hours = self.get_granularity_as_hours(dt_index)
        return abs(actual_hours - expected_hours) < 1e-10


if __name__ == '__main__':
    analyzer = TimeSeriesGranularityAnalyzer(strict_mode=False)

    test_indices = {
        'hourly': pd.date_range('2024-01-01', periods=96, freq='h'),
        'quarter_hourly': pd.date_range('2024-01-01', periods=96, freq='15min'),
        'mixed': (
                list(pd.date_range('2024-01-01', periods=24, freq='h')) +
                list(pd.date_range('2024-01-02', periods=96, freq='15min'))
        ),
    }

    for name, index in test_indices.items():
        print(f"\nAnalyzing {name} index:")
        print(f"Granularity: {analyzer.get_granularity_as_timedelta(index)}")
        print(f"Hours: {analyzer.get_granularity_as_hours(index)}")
        print(f"Minutes: {analyzer.get_granularity_as_minutes(index)}")
        print(f"Series of timedeltas:\n{analyzer.get_granularity_as_series_of_timedeltas(index)}\n")
        print(f"Series of hours:\n{analyzer.get_granularity_as_series_of_hours(index)}\n")
        print(f"Series of minutes:\n{analyzer.get_granularity_as_series_of_minutes(index)}\n")

    # Demonstrating strict mode
    analyzer.strict_mode = True
    try:
        analyzer.get_granularity_as_timedelta(test_indices['mixed'])
    except GranularityError as e:
        print(f"Strict mode error caught: {e}")
