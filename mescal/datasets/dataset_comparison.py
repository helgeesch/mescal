from __future__ import annotations

from typing import Generic
from enum import Enum, auto

import pandas as pd

from mescal.enums import ComparisonTypeEnum
from mescal.datasets.dataset import Dataset
from mescal.datasets.dataset_collection import DatasetCollection, DatasetConcatCollection
from mescal.typevars import FlagType, DatasetConfigType, FlagIndexType, DatasetType
from mescal.utils.pandas_utils.is_numeric import pd_is_numeric


class ComparisonAttributesSourceEnum(Enum):
    USE_VARIATION_ATTS = auto()
    USE_REFERENCE_ATTS = auto()
    USE_INTERSECTION_ATTS = auto()


class DatasetComparison(
    Generic[DatasetType, DatasetConfigType, FlagType, FlagIndexType],
    DatasetCollection[DatasetType, DatasetConfigType, FlagType, FlagIndexType]
):
    COMPARISON_ATTRIBUTES_SOURCE = ComparisonAttributesSourceEnum.USE_VARIATION_ATTS
    COMPARISON_NAME_JOIN = ' vs '
    VARIATION_DS_ATT_KEY = 'variation_dataset'
    REFERENCE_DS_ATT_KEY = 'reference_dataset'

    def __init__(
            self,
            variation_dataset: Dataset,
            reference_dataset: Dataset,
            name: str = None,
            attributes: dict = None,
            config: DatasetConfigType = None,
    ):
        name = name or self._get_auto_generated_name(variation_dataset, reference_dataset)

        super().__init__(
            [reference_dataset, variation_dataset],
            name=name,
            attributes=attributes,
            config=config
        )

        self.variation_dataset = variation_dataset
        self.reference_dataset = reference_dataset

    def _get_auto_generated_name(self, variation_dataset: Dataset, reference_dataset: Dataset) -> str:
        return variation_dataset.name + self.COMPARISON_NAME_JOIN + reference_dataset.name

    @property
    def attributes(self) -> dict:
        match self.COMPARISON_ATTRIBUTES_SOURCE:
            case ComparisonAttributesSourceEnum.USE_VARIATION_ATTS:
                atts = self.variation_dataset.attributes.copy()
            case ComparisonAttributesSourceEnum.USE_REFERENCE_ATTS:
                atts = self.reference_dataset.attributes.copy()
            case _:
                atts = super().attributes
        atts[self.VARIATION_DS_ATT_KEY] = self.variation_dataset.name
        atts[self.REFERENCE_DS_ATT_KEY] = self.reference_dataset.name
        return atts

    def fetch(
            self,
            flag: FlagType,
            config: dict | DatasetConfigType = None,
            comparison_type: ComparisonTypeEnum = ComparisonTypeEnum.DELTA,
            replace_unchanged_values_by_nan: bool = False,
            fill_value: float | int | None = 0,
            **kwargs
    ) -> pd.Series | pd.DataFrame:
        return super().fetch(
            flag=flag,
            config=config,
            comparison_type=comparison_type,
            replace_unchanged_values_by_nan=replace_unchanged_values_by_nan,
            fill_value=fill_value,
            **kwargs
        )

    def _fetch(
            self,
            flag: FlagType,
            effective_config: DatasetConfigType,
            comparison_type: ComparisonTypeEnum = ComparisonTypeEnum.DELTA,
            replace_unchanged_values_by_nan: bool = False,
            fill_value: float | int | None = 0,
            **kwargs
    ) -> pd.Series | pd.DataFrame:
        df_var = self.variation_dataset.fetch(flag, effective_config, **kwargs)
        df_ref = self.reference_dataset.fetch(flag, effective_config, **kwargs)

        match comparison_type:
            case ComparisonTypeEnum.VARIATION:
                return self._get_variation_comparison(df_var, df_ref, replace_unchanged_values_by_nan)
            case ComparisonTypeEnum.BOTH:
                return self._get_both_comparison(df_var, df_ref, replace_unchanged_values_by_nan)
            case ComparisonTypeEnum.DELTA:
                return self._get_delta_comparison(df_var, df_ref, replace_unchanged_values_by_nan, fill_value)
        raise ValueError(f"Unsupported comparison_type: {comparison_type}")

    def _values_are_equal(self, val1, val2) -> bool:
        if pd.isna(val1) and pd.isna(val2):
            return True
        try:
            return val1 == val2
        except:
            pass
        try:
            if str(val1) == str(val2):
                return True
        except:
            pass
        return False

    def _get_variation_comparison(
            self,
            df_var: pd.DataFrame,
            df_ref: pd.DataFrame,
            replace_unchanged_values_by_nan: bool
    ) -> pd.DataFrame:
        result = df_var.copy()

        if replace_unchanged_values_by_nan:
            common_indices = df_var.index.intersection(df_ref.index)
            common_columns = df_var.columns.intersection(df_ref.columns)

            for idx in common_indices:
                for col in common_columns:
                    if self._values_are_equal(df_var.loc[idx, col], df_ref.loc[idx, col]):
                        result.loc[idx, col] = float('nan')

        return result

    def _get_both_comparison(
            self,
            df_var: pd.DataFrame,
            df_ref: pd.DataFrame,
            replace_unchanged_values_by_nan: bool
    ) -> pd.DataFrame:
        var_name = self.variation_dataset.name
        ref_name = self.reference_dataset.name

        result = pd.concat([df_var, df_ref], keys=[var_name, ref_name])
        result = result.sort_index(level=1)

        if replace_unchanged_values_by_nan:
            common_indices = df_var.index.intersection(df_ref.index)
            common_columns = df_var.columns.intersection(df_ref.columns)

            for idx in common_indices:
                for col in common_columns:
                    if self._values_are_equal(df_var.loc[idx, col], df_ref.loc[idx, col]):
                        result.loc[(var_name, idx), col] = float('nan')
                        result.loc[(ref_name, idx), col] = float('nan')

        return result

    def _get_delta_comparison(
            self,
            df_var: pd.DataFrame,
            df_ref: pd.DataFrame,
            replace_unchanged_values_by_nan: bool,
            fill_value: float | int | None
    ) -> pd.DataFrame:
        if pd_is_numeric(df_var) and pd_is_numeric(df_ref):
            result = df_var.subtract(df_ref, fill_value=fill_value)

            if replace_unchanged_values_by_nan:
                result = result.replace(0, float('nan'))

            return result

        all_columns = df_var.columns.union(df_ref.columns)
        all_indices = df_var.index.union(df_ref.index)

        result = pd.DataFrame(index=all_indices, columns=all_columns)

        for col in all_columns:
            if col in df_var.columns and col in df_ref.columns:
                var_col = df_var[col]
                ref_col = df_ref[col]

                # Special handling for boolean columns
                if pd.api.types.is_bool_dtype(var_col) and pd.api.types.is_bool_dtype(ref_col):
                    # For booleans, we can mark where they differ
                    common_indices = var_col.index.intersection(ref_col.index)
                    delta = pd.Series(index=all_indices)

                    for idx in common_indices:
                        if var_col.loc[idx] != ref_col.loc[idx]:
                            delta.loc[idx] = f"{var_col.loc[idx]} (was {ref_col.loc[idx]})"
                        elif not replace_unchanged_values_by_nan:
                            delta.loc[idx] = var_col.loc[idx]

                    # Handle indices only in variation
                    for idx in var_col.index.difference(ref_col.index):
                        delta.loc[idx] = f"{var_col.loc[idx]} (new)"

                    # Handle indices only in reference
                    for idx in ref_col.index.difference(var_col.index):
                        delta.loc[idx] = f"DELETED: {ref_col.loc[idx]}"

                    result[col] = delta

                elif pd.api.types.is_numeric_dtype(var_col) and pd.api.types.is_numeric_dtype(ref_col):
                    delta = var_col.subtract(ref_col, fill_value=fill_value)
                    result[col] = delta

                    if replace_unchanged_values_by_nan:
                        result.loc[delta == 0, col] = float('nan')
                else:
                    common_indices = var_col.index.intersection(ref_col.index)
                    var_only_indices = var_col.index.difference(ref_col.index)
                    ref_only_indices = ref_col.index.difference(var_col.index)

                    for idx in common_indices:
                        if not self._values_are_equal(var_col.loc[idx], ref_col.loc[idx]):
                            result.loc[idx, col] = f"{var_col.loc[idx]} (was {ref_col.loc[idx]})"
                        elif not replace_unchanged_values_by_nan:
                            result.loc[idx, col] = var_col.loc[idx]

                    for idx in var_only_indices:
                        result.loc[idx, col] = f"{var_col.loc[idx]} (new)"

                    for idx in ref_only_indices:
                        val = ref_col.loc[idx]
                        if not pd.isna(val):
                            result.loc[idx, col] = f"DELETED: {val}"

            elif col in df_var.columns:
                for idx in df_var.index:
                    result.loc[idx, col] = f"{df_var.loc[idx, col]} (new column)"

            else:  # Column only in reference
                for idx in df_ref.index:
                    val = df_ref.loc[idx, col]
                    if not pd.isna(val):
                        result.loc[idx, col] = f"REMOVED: {val}"

        return result


class DatasetConcatCollectionOfComparisons(
    Generic[DatasetConfigType, FlagType, FlagIndexType],
    DatasetConcatCollection[DatasetComparison, DatasetConfigType, FlagType, FlagIndexType]
):
    @classmethod
    def get_child_dataset_type(cls) -> type[DatasetType]:
        return DatasetComparison

    def fetch(
            self,
            flag: FlagType,
            config: dict | DatasetConfigType = None,
            comparison_type: ComparisonTypeEnum = ComparisonTypeEnum.DELTA,
            replace_unchanged_values_by_nan: bool = False,
            fill_value: float | int | None = 0,
            **kwargs
    ) -> pd.Series | pd.DataFrame:
        return super().fetch(
            flag=flag,
            config=config,
            comparison_type=comparison_type,
            replace_unchanged_values_by_nan=replace_unchanged_values_by_nan,
            fill_value=fill_value,
            **kwargs
        )
