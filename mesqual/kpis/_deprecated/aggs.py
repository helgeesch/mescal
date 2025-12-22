from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

import numpy as np
import pandas as pd

from mesqual.units import Units
from mesqual.energy_data_handling.granularity_analyzer import TimeSeriesGranularityAnalyzer

if TYPE_CHECKING:
    from mesqual._kpis_deprecated.kpi_base import KPI_VALUE_TYPES


@dataclass
class Aggregation:
    name: str
    agg: Callable[[pd.DataFrame], KPI_VALUE_TYPES]
    unit: Units.Unit = None

    def __call__(self, df: pd.DataFrame) -> KPI_VALUE_TYPES:
        return self.agg(df)

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other) -> bool:
        return isinstance(other, Aggregation) and self.name == other.name


def _ensure_frame_format(pd_object: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(pd_object, pd.Series):
        return pd_object.to_frame('value')
    return pd_object


def _annualized_sum(df: pd.Series | pd.DataFrame) -> float:
    total_hours = TimeSeriesGranularityAnalyzer(strict_mode=False).get_granularity_as_series_of_hours(df.index).sum()
    tmp = _ensure_frame_format(df).sum(axis=1)
    return tmp.sum() / total_hours * 8760


def _daily_sum(df: pd.Series | pd.DataFrame) -> float:
    total_hours = TimeSeriesGranularityAnalyzer(strict_mode=False).get_granularity_as_series_of_hours(df.index).sum()
    tmp = _ensure_frame_format(df).sum(axis=1)
    return tmp.sum() / total_hours * 24


def _sum_df_to_series(df: pd.DataFrame) -> pd.Series:
    s = df.sum(axis=1)
    s[df.isna().all(axis=1)] = np.nan
    return s


class Aggregations:
    Total = Aggregation('Total', lambda df: _sum_df_to_series(_ensure_frame_format(df)).sum())
    Sum = Aggregation('Sum', lambda df: _sum_df_to_series(_ensure_frame_format(df)).sum())
    AnnualizedSum = Aggregation('AnnualizedSum', lambda df: _annualized_sum(df))
    DailySum = Aggregation('DailySum', lambda df: _daily_sum(df))
    Max = Aggregation('Max', lambda df: _sum_df_to_series(_ensure_frame_format(df)).max())
    Mean = Aggregation('Mean', lambda df: _sum_df_to_series(_ensure_frame_format(df)).mean())
    Min = Aggregation('Min', lambda df: _sum_df_to_series(_ensure_frame_format(df)).min())
    AbsSum = Aggregation('AbsSum', lambda df: _sum_df_to_series(_ensure_frame_format(df).abs()).sum())
    AbsMax = Aggregation('AbsMax', lambda df: _sum_df_to_series(_ensure_frame_format(df).abs()).max())
    AbsMean = Aggregation('AbsMean', lambda df: _sum_df_to_series(_ensure_frame_format(df).abs()).mean())
    AbsMin = Aggregation('AbsMin', lambda df: _sum_df_to_series(_ensure_frame_format(df).abs()).min())
    SumGeqZero = Aggregation('SumGeqZero', lambda df: _sum_df_to_series(_ensure_frame_format(df).clip(0, None)).sum())
    SumLeqZero = Aggregation('SumLeqZero', lambda df: _sum_df_to_series(_ensure_frame_format(df).clip(None, 0)).sum())
    MeanGeqZero = Aggregation('MeanGeqZero', lambda df: _sum_df_to_series(_ensure_frame_format(df).clip(0, None)).mean())
    MeanLeqZero = Aggregation('MeanLeqZero', lambda df: _sum_df_to_series(_ensure_frame_format(df).clip(None, 0)).mean())
    MTUsWithNaN = Aggregation('MTUsWithNaN', lambda df: _ensure_frame_format(df).isna().any(axis=1).sum(), Units.MTU)
    MTUsNonZero = Aggregation('MTUsNonZero', lambda df: ((_ensure_frame_format(df) != 0) & ~_ensure_frame_format(df).isna()).any(axis=1).sum(), Units.MTU)
    MTUsEqZero = Aggregation('MTUsEqZero', lambda df: (_ensure_frame_format(df) == 0).any(axis=1).sum(), Units.MTU)
    MTUsAboveZero = Aggregation('MTUsAboveZero', lambda df: (_ensure_frame_format(df) > 0).any(axis=1).sum(), Units.MTU)
    MTUsBelowZero = Aggregation('MTUsBelowZero', lambda df: (_ensure_frame_format(df) < 0).any(axis=1).sum(), Units.MTU)
    MTUsAboveX = lambda x: Aggregation(f'MTUsAbove{x}', lambda df: (_ensure_frame_format(df) > x).any(axis=1).sum(), Units.MTU)
    MTUsBelowX = lambda x: Aggregation(f'MTUsBelow{x}', lambda df: (_ensure_frame_format(df) < x).any(axis=1).sum(), Units.MTU)


@dataclass
class OperationOfTwoValues:
    name: str
    agg: Callable[[KPI_VALUE_TYPES, KPI_VALUE_TYPES], KPI_VALUE_TYPES]
    unit: Units.Unit = None

    def __call__(self, variation_value: KPI_VALUE_TYPES, reference_value: KPI_VALUE_TYPES) -> KPI_VALUE_TYPES:
        return self.agg(variation_value, reference_value)

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)


class ValueComparison(OperationOfTwoValues):
    pass


class ValueComparisons:
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
    pass


class ArithmeticValueOperations:
    Product = ArithmeticValueOperation("Product", lambda var, ref: var * ref)
    Division = ArithmeticValueOperation("Division", lambda var, ref: np.inf * np.sign(var) if ref == 0 else var / ref)
    Share = ArithmeticValueOperation(
        "Share",
        lambda var, ref: np.inf * np.sign(var) if ref == 0 else var / ref * 100,
        Units.percent
    )
    Sum = ArithmeticValueOperation("Sum", lambda var, ref: var + ref)
    Diff = ArithmeticValueOperation("Diff", lambda var, ref: var - ref)
    Delta = ArithmeticValueOperation("Delta", lambda var, ref: var - ref)
