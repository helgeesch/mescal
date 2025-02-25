from __future__ import annotations

from typing import Generic

import pandas as pd

from mescal.datasets.dataset import Dataset
from mescal.datasets.dataset_collection import DatasetCollection, DatasetConcatCollection
from mescal.typevars import FlagType, DatasetConfigType, FlagIndexType, DatasetType
from mescal.utils.pandas_utils.is_numeric import pd_is_numeric


class DatasetComparison(
    Generic[DatasetType, DatasetConfigType, FlagType, FlagIndexType],
    DatasetCollection[DatasetType, DatasetConfigType, FlagType, FlagIndexType]
):
    """
    Takes two Datasets (variation and reference) and fetch method will return the delta between the two (var-ref).
    """
    def __init__(
            self,
            variation_dataset: Dataset,
            reference_dataset: Dataset,
            name: str = None,
            attributes: dict = None,
            config: DatasetConfigType = None,
    ):
        if name is None:
            name = variation_dataset.name + ' vs ' + reference_dataset.name

        super().__init__(
            [reference_dataset, variation_dataset],
            name=name,
            attributes=attributes,
            config=config
        )

        self.variation_dataset = variation_dataset
        self.reference_dataset = reference_dataset

    def fetch(
            self,
            flag: FlagType,
            config: dict | DatasetConfigType = None,
            fill_value: float | int | None = 0,
            **kwargs
    ) -> pd.Series | pd.DataFrame:
        return super().fetch(flag, config, fill_value=fill_value, **kwargs)

    def _fetch(
            self,
            flag: FlagType,
            effective_config: DatasetConfigType,
            fill_value: float | int | None = 0,
            **kwargs
    ) -> pd.Series | pd.DataFrame:
        df_var: pd.DataFrame = self.variation_dataset.fetch(flag, effective_config, **kwargs)
        df_ref: pd.DataFrame = self.reference_dataset.fetch(flag, effective_config, **kwargs)

        if pd_is_numeric(df_var) and pd_is_numeric(df_ref):
            return df_var.subtract(df_ref, fill_value=fill_value)

        # TODO: implement other kinds of comparisons (can be an enum)

        raise NotImplementedError


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
            fill_value: float | int | None = 0,
            **kwargs
    ) -> pd.Series | pd.DataFrame:
        return super().fetch(flag, config, fill_value=fill_value, **kwargs)
