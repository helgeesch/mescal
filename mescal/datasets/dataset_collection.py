from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterator, Callable, Generic, Type, Iterable, TYPE_CHECKING

import pandas as pd

from mescal.datasets.dataset import Dataset
from mescal.flag.flag_index import FlagIndex
from mescal.utils.pandas_utils.is_numeric import pd_is_numeric
from mescal.utils.logging import get_logger
from mescal.utils.pandas_utils.combine_df import combine_dfs
from mescal.utils.set_aggregations import nested_union
from mescal.utils.intersect_dicts import get_intersection_of_dicts
from mescal.typevars import DatasetType, DatasetConfigType, FlagType, FlagIndexType

if TYPE_CHECKING:
    from mescal.kpis.kpi_base import KPIFactory
    from mescal.kpis.kpi_collection import KPICollection
    from mescal.databases.database import Database

logger = get_logger(__name__)


class DatasetCollection(
    Generic[DatasetType, DatasetConfigType, FlagType, FlagIndexType],
    Dataset[DatasetConfigType, FlagType, FlagIndexType],
    ABC
):
    """
    Abstract class to collect multiple Dataset instances
    and handle them according to a specific logic.
    Inherits all methods / functionalities from Dataset.
    """

    def __init__(
            self,
            datasets: list[DatasetType] = None,
            name: str = None,
            parent_dataset: Dataset = None,
            flag_index: FlagIndex = None,
            attributes: dict = None,
            database: Database = None,
            config: DatasetConfigType = None
    ):
        super().__init__(
            name=name,
            parent_dataset=parent_dataset,
            flag_index=flag_index,
            attributes=attributes,
            database=database,
            config=config,
        )
        self.datasets: list[DatasetType] = datasets if datasets else []

    @property
    def flag_index(self) -> FlagIndex:
        from mescal.flag.flag_index import EmptyFlagIndex
        if (self._flag_index is None) or isinstance(self._flag_index, EmptyFlagIndex):
            from mescal.utils.check_all_same import all_same_object
            if all_same_object(ds.flag_index for ds in self.datasets) and len(self.datasets):
                return self.get_dataset().flag_index
        return self._flag_index

    @property
    def attributes(self) -> pd.Series:
        atts = self._attributes.copy()
        child_dataset_atts = [ds.attributes.to_dict() for ds in self.datasets]
        attributes_that_all_childs_have_in_common = get_intersection_of_dicts(child_dataset_atts)
        for key in atts.keys():
            attributes_that_all_childs_have_in_common.pop(key, None)
        atts.update(attributes_that_all_childs_have_in_common)
        return pd.Series(atts, name=self.name)

    @property
    def attributes_df(self) -> pd.DataFrame:
        return pd.concat(
            {ds.name: ds.attributes for ds in self.datasets},
            axis=1,
            names=['dataset']
        ).rename_axis('attribute').T

    def get_kpi_df(self, **kwargs) -> pd.DataFrame:
        # TODO: concat axis
        # TODO: uniform order of magnitude per KPI / KPI category
        return pd.concat(
            {ds.name: ds.get_kpi_series(**kwargs) for ds in self.datasets},
            axis=1,
            names=['dataset']
        )

    def get_merged_kpi_collection(self, deep: bool = True) -> 'KPICollection':
        from mescal.kpis.kpi_collection import KPICollection
        all_kpis = set()
        for ds in self.datasets:
            for kpi in ds.kpi_collection:
                all_kpis.add(kpi)
            if deep and isinstance(ds, DatasetCollection):
                for kpi in ds.get_merged_kpi_collection(deep=deep):
                    all_kpis.add(kpi)

        return KPICollection(all_kpis)

    def add_kpis_to_all_sub_datasets(self, kpis: Iterable[KPIFactory]):
        for kpi in kpis:
            self.add_kpi_to_all_sub_datasets(kpi)

    def add_kpi_to_all_sub_datasets(self, kpi: KPIFactory):
        for ds in self.datasets:
            ds.add_kpi(kpi)

    def clear_kpi_collection_for_all_sub_datasets(self, deep: bool = True):
        for ds in self.datasets:
            ds.clear_kpi_collection()
            if deep and isinstance(ds, DatasetCollection):
                ds.clear_kpi_collection_for_all_sub_datasets(deep=deep)

    @abstractmethod
    def _fetch(
            self,
            flag: FlagType,
            effective_config: DatasetConfigType,
            **kwargs
    ) -> pd.Series | pd.DataFrame:
        pass

    def flag_is_accepted(self, flag: FlagType) -> bool:
        return any(ds.flag_is_accepted(flag) for ds in self.datasets)

    @property
    def accepted_flags(self) -> set[FlagType]:
        return nested_union([ds.accepted_flags for ds in self.datasets])

    def _required_flags_for_flag(self, flag: FlagType) -> set[FlagType]:
        return nested_union([ds.accepted_flags for ds in self.datasets])

    def get_dataset(self, key: str = None) -> DatasetType:
        if key is None:
            if not self.datasets:
                raise ValueError("No datasets available")
            return self.datasets[0]

        for ds in self.datasets:
            if ds.name == key:
                return ds

        raise KeyError(f"Dataset with name '{key}' not found")

    def add_datasets(self, datasets: Iterable[DatasetType]):
        for ds in datasets:
            self.add_dataset(ds)

    def add_dataset(self, dataset: DatasetType):
        if not isinstance(dataset, self.get_child_dataset_type()):
            raise TypeError(f"Can only add data sets of type {self.get_child_dataset_type().__name__}.")

        for i, existing in enumerate(self.datasets):
            if existing.name == dataset.name:
                logger.warning(
                    f"Dataset {self.name}: "
                    f"dataset {dataset.name} already in this collection. Replacing it."
                )
                self.datasets[i] = dataset
                return

        self.datasets.append(dataset)

    @classmethod
    def get_child_dataset_type(cls) -> type[DatasetType]:
        return Dataset


class DatasetLinkCollection(
    Generic[DatasetType, DatasetConfigType, FlagType, FlagIndexType],
    DatasetCollection[DatasetType, DatasetConfigType, FlagType, FlagIndexType]
):
    """
    Links multiple Dataset instances so that:
        - the parent-Dataset accepts flags of all child-Datasets and automatically returns the data
          from the child-Dataset one that accepts the flag.
        - the child-Dataset instances have access to the parent-Dataset so that they can fetch from other,
          e.g. child_ds.parent_dataset.fetch(...).
    """

    def __init__(
            self,
            datasets: list[DatasetType],
            name: str = None,
            parent_dataset: Dataset = None,
            flag_index: FlagIndex = None,
            attributes: dict = None,
            database: Database = None,
            config: DatasetConfigType = None,
    ):
        super().__init__(
            datasets=datasets,
            name=name,
            parent_dataset=parent_dataset,
            flag_index=flag_index,
            attributes=attributes,
            database=database,
            config=config,
        )
        self._warn_if_flags_overlap()

    def _fetch(self, flag: FlagType, effective_config: DatasetConfigType, **kwargs) -> pd.Series | pd.DataFrame:
        for ds in self.datasets:
            if ds.flag_is_accepted(flag):
                return ds.fetch(flag, effective_config, **kwargs)
        raise KeyError(f"Key '{flag}' not recognized by any of the linked Datasets.")

    def _warn_if_flags_overlap(self):
        from collections import Counter

        accepted_flags = list()
        for ds in self.datasets:
            accepted_flags += list(ds.accepted_flags)

        counts = Counter(accepted_flags)
        duplicates = {k: v for k, v in counts.items() if v > 1}
        if any(duplicates.values()):
            logger.warning(
                f"Dataset {self.name}: "
                f"The following keys have multiple Dataset sources: {duplicates.keys()}. \n"
                f"Only the first one will be used! This might lead to unexpected behavior. \n"
                f"A potential reason could be the use of an inappropriate DatasetCollection Type."
            )


class DatasetMergeCollection(
    Generic[DatasetType, DatasetConfigType, FlagType, FlagIndexType],
    DatasetCollection[DatasetType, DatasetConfigType, FlagType, FlagIndexType]
):
    """
    Fetch method will merge fragmented Datasets for same flag, e.g.:
        - fragmented simulation runs, e.g. CW1, CW2, CW3, CWn.
        - fragmented data sources, e.g. mapping from Excel file with model from simulation platform.
    """
    def __init__(
            self,
            datasets: list[DatasetType],
            name: str = None,
            parent_dataset: Dataset = None,
            flag_index: FlagIndex = None,
            attributes: dict = None,
            database: Database = None,
            config: DatasetConfigType = None,
            keep_first: bool = True,
    ):
        super().__init__(
            datasets=datasets,
            name=name,
            parent_dataset=parent_dataset,
            flag_index=flag_index,
            attributes=attributes,
            database=database,
            config=config,
        )
        self.keep_first = keep_first

    def _fetch(self, flag: FlagType, effective_config: DatasetConfigType, **kwargs) -> pd.Series | pd.DataFrame:
        data_frames = []
        for ds in self.datasets:
            if ds.flag_is_accepted(flag):
                data_frames.append(ds.fetch(flag, effective_config, **kwargs))

        if not data_frames:
            raise KeyError(f"Flag '{flag}' not recognized by any of the datasets.")

        from mescal.utils.pandas_utils.combine_df import combine_dfs
        df = combine_dfs(data_frames, keep_first=self.keep_first)
        return df


class DatasetConcatCollection(
    Generic[DatasetType, DatasetConfigType, FlagType, FlagIndexType],
    DatasetCollection[DatasetType, DatasetConfigType, FlagType, FlagIndexType]
):
    """
    Fetch method will return a concatenation of all child-Datasets with an additional Index-level.
    """
    def __init__(
            self,
            datasets: list[DatasetType],
            name: str = None,
            parent_dataset: Dataset = None,
            flag_index: FlagIndex = None,
            attributes: dict = None,
            database: Database = None,
            config: DatasetConfigType = None,
            default_concat_axis: int = 1,
            concat_top: bool = True,
            concat_level_name: str = None,
    ):
        super().__init__(
            datasets=datasets,
            name=name,
            parent_dataset=parent_dataset,
            flag_index=flag_index,
            attributes=attributes,
            database=database,
            config=config,
        )
        super().__init__(datasets=datasets, name=name)
        self.default_concat_axis = default_concat_axis
        self.concat_top = concat_top
        self.concat_level_name = concat_level_name if concat_level_name is not None else 'dataset'

    def get_kpi_df(self, **kwargs) -> pd.DataFrame:
        # TODO: concat axis
        # TODO: uniform order of magnitude per KPI / KPI category
        return pd.concat(
            {ds.name: ds.get_kpi_series(**kwargs) for ds in self.datasets},
            axis=1,
            names=[self.concat_level_name]
        )

    def fetch_merged(
            self,
            flag: FlagType,
            config: dict | DatasetConfigType = None,
            keep_first: bool = True,
            **kwargs
    ) -> pd.Series | pd.DataFrame:
        """Fetch method that merges dataframes from all child datasets, similar to DatasetMergeCollection."""
        temp_merge_collection = DatasetMergeCollection(
            datasets=self.datasets,
            name=f"{self.name}_temp_merge",
            keep_first=keep_first
        )
        return temp_merge_collection.fetch(flag, config, **kwargs)

    def _fetch(
            self,
            flag: FlagType,
            effective_config: DatasetConfigType,
            concat_axis: int = None,
            **kwargs
    ) -> pd.Series | pd.DataFrame:
        if concat_axis is None:
            concat_axis = self.default_concat_axis

        dfs = {}
        for ds in self.datasets:
            if ds.flag_is_accepted(flag):
                dfs[ds.name] = ds.fetch(flag, effective_config, **kwargs)

        if not dfs:
            raise KeyError(f"Flag '{flag}' not recognized by any of the datasets.")

        df0 = list(dfs.values())[0]
        if not all(len(df.axes) == len(df0.axes) for df in dfs.values()):
            raise NotImplementedError(f'Axes lengths do not match between dfs.')

        for ax in range(len(df0.axes)):
            if not all(set(df.axes[ax].names) == set(df0.axes[ax].names) for df in dfs.values()):
                raise NotImplementedError(f'Axes names do not match between dfs.')

        df = pd.concat(dfs, join='outer', axis=concat_axis, names=[self.concat_level_name])

        if not self.concat_top:
            ax = df.axes[concat_axis]
            df.axes[concat_axis] = ax.reorder_levels([ax.nlevels - 1] + list(range(ax.nlevels - 1)))

        return df


class DatasetSumCollection(
    Generic[DatasetType, DatasetConfigType, FlagType, FlagIndexType],
    DatasetCollection[DatasetType, DatasetConfigType, FlagType, FlagIndexType]
):
    def _fetch(self, flag: FlagType, effective_config: DatasetConfigType, **kwargs) -> pd.Series | pd.DataFrame:
        data: list[pd.Series | pd.DataFrame] = []
        for ds in self.datasets:
            if ds.flag_is_accepted(flag):
                data.append(ds.fetch(flag, effective_config, **kwargs))
        if not data:
            raise KeyError(f"Flag '{flag}' not recognized by any of the linked Datasets.")
        
        if all(pd_is_numeric(d) for d in data):
            import numpy as np
            return np.sum(data)

        raise NotImplementedError
