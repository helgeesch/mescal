import numpy as np
import pandas as pd


class TimeSeriesGapHandler:
    def __init__(self, max_gap_in_minutes: float = 60):
        self._max_gap_in_minutes = max_gap_in_minutes

    def insert_nans_at_gaps(self, data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        diffs = data.index.to_series().diff()
        mask = diffs > pd.Timedelta(minutes=self._max_gap_in_minutes)
        gap_indices = np.where(mask)[0]
        new_timestamps = data.index[gap_indices - 1] + pd.Timedelta(minutes=self._max_gap_in_minutes)

        if isinstance(data, pd.Series):
            new_values = pd.Series(np.nan, index=new_timestamps)
        else:
            new_values = pd.DataFrame(np.nan, index=new_timestamps, columns=data.columns)

        return pd.concat([data, new_values]).sort_index()


if __name__ == '__main__':
    dates = pd.date_range('2024-01-01', periods=5, freq='30min')
    dates = dates.delete(2)
    values = np.random.rand(len(dates))
    sample_series = pd.Series(values, index=dates)

    handler = TimeSeriesGapHandler(max_gap_in_minutes=30)
    result = handler.insert_nans_at_gaps(sample_series)
    print(result)
