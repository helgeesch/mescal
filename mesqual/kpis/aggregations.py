from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from mesqual.units import Units
from mesqual.energy_data_handling.granularity_analyzer import TimeSeriesGranularityAnalyzer


@dataclass
class Aggregation:
    """
    Aggregation function for batch KPI computation.

    Unlike the old system, these aggregations:

        - Operate on entire DataFrames
        - Return Series with one value per column (one per object)
        - Enable batch computation across all objects

    Attributes:
        name: Human-readable name of the aggregation
        agg: Function that takes DataFrame and returns Series
        unit: Optional unit override (e.g., MTU for time-based counts)
    """
    name: str
    agg: Callable[[pd.DataFrame], pd.Series]
    unit: Units.Unit = None

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        """Apply aggregation to DataFrame, return Series with one value per column."""
        return self.agg(df)

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other) -> bool:
        return isinstance(other, Aggregation) and self.name == other.name


def _ensure_frame_format(pd_object: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """
    Convert Series to DataFrame if needed.

    Args:
        pd_object: Series or DataFrame to ensure is DataFrame format

    Returns:
        DataFrame (either original or converted from Series)
    """
    if isinstance(pd_object, pd.Series):
        return pd_object.to_frame('value')
    return pd_object


def _column_wise_sum(df: pd.DataFrame) -> pd.Series:
    """
    Sum each column, preserving NaN handling.

    If all values in a column are NaN, result is NaN.
    Otherwise, sum ignoring NaN values.

    Args:
        df: DataFrame to sum column-wise

    Returns:
        Series with sum per column
    """
    result = df.sum(axis=0)
    result[df.isna().all(axis=0)] = np.nan
    return result


def _annualized_sum_batch(df: pd.DataFrame) -> pd.Series:
    """
    Annualized sum for each column.

    Computes the sum over the time period and scales to annual (8760 hours).

    Args:
        df: DataFrame with DatetimeIndex

    Returns:
        Series with annualized sum per column
    """
    total_hours = TimeSeriesGranularityAnalyzer(strict_mode=False).get_granularity_as_series_of_hours(df.index).sum()
    return df.sum(axis=0) / total_hours * 8760


def _daily_sum_batch(df: pd.DataFrame) -> pd.Series:
    """
    Daily sum for each column.

    Computes the sum over the time period and scales to daily (24 hours).

    Args:
        df: DataFrame with DatetimeIndex

    Returns:
        Series with daily sum per column
    """
    total_hours = TimeSeriesGranularityAnalyzer(strict_mode=False).get_granularity_as_series_of_hours(df.index).sum()
    return df.sum(axis=0) / total_hours * 24


class Aggregations:
    """
    Standard aggregations for batch KPI computation.

    All aggregations operate column-wise on DataFrames and return
    a Series with one value per column.

    Example:

        >>> df = pd.DataFrame({'obj_1': [10, 20, 30], 'obj_2': [15, 25, 35]})
        >>> Aggregations.Mean(df)
            pd.Series({'obj_1': 20.0, 'obj_2': 25.0})
    """

    # Basic statistical aggregations
    Sum = Aggregation(
        'Sum',
        lambda df: _column_wise_sum(_ensure_frame_format(df))
    )
    Total = Aggregation(
        'Total',
        lambda df: _column_wise_sum(_ensure_frame_format(df))
    )
    Mean = Aggregation(
        'Mean',
        lambda df: _ensure_frame_format(df).mean(axis=0)
    )
    Max = Aggregation(
        'Max',
        lambda df: _ensure_frame_format(df).max(axis=0)
    )
    Min = Aggregation(
        'Min',
        lambda df: _ensure_frame_format(df).min(axis=0)
    )

    # Time-based aggregations
    AnnualizedSum = Aggregation('AnnualizedSum', _annualized_sum_batch)
    DailySum = Aggregation('DailySum', _daily_sum_batch)

    # Absolute value aggregations
    AbsSum = Aggregation(
        'AbsSum',
        lambda df: _ensure_frame_format(df).abs().sum(axis=0)
    )
    AbsMax = Aggregation(
        'AbsMax',
        lambda df: _ensure_frame_format(df).abs().max(axis=0)
    )
    AbsMean = Aggregation(
        'AbsMean',
        lambda df: _ensure_frame_format(df).abs().mean(axis=0)
    )
    AbsMin = Aggregation(
        'AbsMin',
        lambda df: _ensure_frame_format(df).abs().min(axis=0)
    )

    # Clipped aggregations (positive/negative only)
    SumGeqZero = Aggregation(
        'SumGeqZero',
        lambda df: _ensure_frame_format(df).clip(0, None).sum(axis=0)
    )
    SumLeqZero = Aggregation(
        'SumLeqZero',
        lambda df: _ensure_frame_format(df).clip(None, 0).sum(axis=0)
    )
    MeanGeqZero = Aggregation(
        'MeanGeqZero',
        lambda df: _ensure_frame_format(df).clip(0, None).mean(axis=0)
    )
    MeanLeqZero = Aggregation(
        'MeanLeqZero',
        lambda df: _ensure_frame_format(df).clip(None, 0).mean(axis=0)
    )

    # MTU (Market Time Unit) aggregations - count timesteps
    MTUsWithNaN = Aggregation(
        'MTUsWithNaN',
        lambda df: _ensure_frame_format(df).isna().sum(axis=0),
        Units.MTU
    )
    MTUsNonZero = Aggregation(
        'MTUsNonZero',
        lambda df: ((_ensure_frame_format(df) != 0) & ~_ensure_frame_format(df).isna()).sum(axis=0),
        Units.MTU
    )
    MTUsEqZero = Aggregation(
        'MTUsEqZero',
        lambda df: (_ensure_frame_format(df) == 0).sum(axis=0),
        Units.MTU
    )
    MTUsAboveZero = Aggregation(
        'MTUsAboveZero',
        lambda df: (_ensure_frame_format(df) > 0).sum(axis=0),
        Units.MTU
    )
    MTUsBelowZero = Aggregation(
        'MTUsBelowZero',
        lambda df: (_ensure_frame_format(df) < 0).sum(axis=0),
        Units.MTU
    )

    # Parameterized MTU aggregations
    @staticmethod
    def MTUsAboveX(x: float) -> Aggregation:
        """
        Count timesteps where value is above threshold.

        Args:
            x: Threshold value

        Returns:
            Aggregation that counts timesteps > x per column
        """
        return Aggregation(
            f'MTUsAbove{x}',
            lambda df: (_ensure_frame_format(df) > x).sum(axis=0),
            Units.MTU
        )

    @staticmethod
    def MTUsBelowX(x: float) -> Aggregation:
        """
        Count timesteps where value is below threshold.

        Args:
            x: Threshold value

        Returns:
            Aggregation that counts timesteps < x per column
        """
        return Aggregation(
            f'MTUsBelow{x}',
            lambda df: (_ensure_frame_format(df) < x).sum(axis=0),
            Units.MTU
        )


@dataclass
class OperationOfTwoValues:
    """
    Operation that combines two scalar values.

    Used for comparison KPIs (e.g., difference between scenarios)
    and arithmetic operations between KPI values.

    Attributes:
        name: Human-readable name of the operation
        agg: Function that takes two scalars and returns a scalar
        unit: Optional unit override
    """
    name: str
    agg: Callable[[float | int, float | int], float | int]
    unit: Units.Unit = None

    def __call__(self, variation_value: float | int, reference_value: float | int) -> float | int:
        """Apply operation to two values."""
        return self.agg(variation_value, reference_value)

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other) -> bool:
        return isinstance(other, OperationOfTwoValues) and self.name == other.name


class ValueComparison(OperationOfTwoValues):
    """Comparison operation between two KPI values (e.g., difference, percentage change)."""
    pass


class ValueComparisons:
    """
    Standard comparison operations for KPIs.

    Used to compare variation vs reference scenarios.

    Example:

        >>> ref_value = 100
        >>> var_value = 120
        >>> ValueComparisons.Increase(var_value, ref_value)  # 20
        >>> ValueComparisons.PercentageIncrease(var_value, ref_value)  # 20.0
    """

    Increase = ValueComparison("Increase", lambda var, ref: var - ref)
    Decrease = ValueComparison("Decrease", lambda var, ref: ref - var)
    PercentageIncrease = ValueComparison(
        "PercentageIncrease",
        lambda var, ref: np.inf * np.sign(var) if ref == 0 else (var - ref) / ref * 100,
        Units.percent
    )
    PercentageDecrease = ValueComparison(
        "PercentageDecrease",
        lambda var, ref: -1 * np.inf * np.sign(var) if ref == 0 else (ref - var) / ref * 100,
        Units.percent
    )
    Share = ValueComparison(
        "Share",
        lambda var, ref: np.inf * np.sign(var) if ref == 0 else var / ref * 100,
        Units.percent
    )
    Delta = ValueComparison("Delta", lambda var, ref: var - ref)
    Diff = ValueComparison("Diff", lambda var, ref: var - ref)


class ArithmeticValueOperation(OperationOfTwoValues):
    """Arithmetic operation between two KPI values (e.g., sum, product)."""
    pass


class ArithmeticValueOperations:
    """
    Standard arithmetic operations for KPIs.

    Used to derive new KPI values from existing ones.

    Example:

        >>> value1 = 100
        >>> value2 = 50
        >>> ArithmeticValueOperations.Sum(value1, value2)  # 150
        >>> ArithmeticValueOperations.Division(value1, value2)  # 2.0
    """

    Product = ArithmeticValueOperation("Product", lambda var, ref: var * ref)
    Division = ArithmeticValueOperation(
        "Division",
        lambda var, ref: np.inf * np.sign(var) if ref == 0 else var / ref
    )
    Share = ArithmeticValueOperation(
        "Share",
        lambda var, ref: np.inf * np.sign(var) if ref == 0 else var / ref * 100,
        Units.percent
    )
    Sum = ArithmeticValueOperation("Sum", lambda var, ref: var + ref)
    Diff = ArithmeticValueOperation("Diff", lambda var, ref: var - ref)
    Delta = ArithmeticValueOperation("Delta", lambda var, ref: var - ref)
