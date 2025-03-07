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


def _never_skip_ds_condition(ds: Dataset) -> bool:
    return False


def skip_ds_if_flag_not_accepted_condition(flag: FlagType) -> Callable[[Dataset], bool]:
    return lambda ds: not ds.flag_is_accepted(flag)


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
        datasets = datasets if datasets else []
        self.datasets: dict[str, DatasetType] = {ds.name: ds for ds in datasets}

    @property
    def flag_index(self) -> FlagIndex:
        from mescal.flag.flag_index import EmptyFlagIndex
        if (self._flag_index is None) or isinstance(self._flag_index, EmptyFlagIndex):
            from mescal.utils.check_all_same import all_same_object
            if all_same_object(ds.flag_index for ds in self.dataset_iterator) and len(self.datasets):
                return self.get_dataset().flag_index
        return self._flag_index

    @property
    def attributes(self) -> pd.Series:
        atts = self._attributes.copy()
        child_dataset_atts = [ds.attributes.to_dict() for ds in self.datasets.values()]
        attributes_that_all_childs_have_in_common = get_intersection_of_dicts(child_dataset_atts)
        for key in atts.keys():
            attributes_that_all_childs_have_in_common.pop(key, None)
        atts.update(attributes_that_all_childs_have_in_common)
        return pd.Series(atts, name=self.name)

    @property
    def attributes_df(self) -> pd.DataFrame:
        return pd.concat(
            {ds.name: ds.attributes for ds in self.dataset_iterator},
            axis=1,
            names=['dataset']
        ).rename_axis('attribute').T

    def get_kpi_df(self, **kwargs) -> pd.DataFrame:
        # TODO: concat axis
        # TODO: uniform order of magnitude per KPI / KPI category
        return pd.concat(
            {ds.name: ds.get_kpi_series(**kwargs) for ds in self.dataset_iterator},
            axis=1,
            names=['dataset']
        )

    def get_merged_kpi_collection(self, deep: bool = True) -> 'KPICollection':
        from mescal.kpis.kpi_collection import KPICollection
        all_kpis = set()
        for ds in self.dataset_iterator:
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
        for ds in self.dataset_iterator:
            ds.add_kpi(kpi)

    def clear_kpi_collection_for_all_sub_datasets(self, deep: bool = True):
        for ds in self.dataset_iterator:
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
        return any(ds.flag_is_accepted(flag) for ds in self.dataset_iterator)

    @property
    def accepted_flags(self) -> set[FlagType]:
        return nested_union([ds.accepted_flags for ds in self.dataset_iterator])

    def _required_flags_for_flag(self, flag: FlagType) -> set[FlagType]:
        return nested_union([ds.accepted_flags for ds in self.dataset_iterator])

    @property
    def dataset_iterator(self) -> Iterator[DatasetType]:
        for ds in self.datasets.values():
            yield ds

    def get_dataset(self, key: str = None) -> DatasetType:
        if key is None:
            key = list(self.datasets.keys())[0]
        return self.datasets[key]

    def add_datasets(self, datasets: Iterable[DatasetType]):
        for ds in datasets:
            self.add_dataset(ds)

    def add_dataset(self, dataset: DatasetType):
        if not isinstance(dataset, self.get_child_dataset_type()):
            raise TypeError(f"Can only add data sets of type {self.get_child_dataset_type().__name__}.")
        if dataset.name not in self.datasets:
            self.datasets[dataset.name] = dataset
        else:
            logger.warning(
                f"Dataset {self.name}: "
                f"dataset {dataset.name} already in {type(self).__name__}. Not added again."
            )

    @classmethod
    def get_child_dataset_type(cls) -> type[DatasetType]:
        return Dataset


class DatasetLinkCollection(
    Generic[DatasetType, DatasetConfigType, FlagType, FlagIndexType],
    DatasetCollection[DatasetType, DatasetConfigType, FlagType, FlagIndexType]
):
    """
    Links multiple Dataset instances so:
        - the parent Dataset accepts flags of all child Datasets.
        - the child Dataset instances can fetch from each other.
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
        for ds in self.dataset_iterator:
            if ds.flag_is_accepted(flag):
                return ds.fetch(flag, effective_config, **kwargs)
        raise KeyError(f"Key '{flag}' not recognized by any of the linked Datasets.")

    def _warn_if_flags_overlap(self):
        from collections import Counter

        accepted_flags = list()
        for ds in self.dataset_iterator:
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
        df = self._combine_dfs(
            get_df_from_dataset_method=lambda ds: ds.fetch(flag, effective_config, **kwargs),
            keep_first=self.keep_first,
            skip_ds_condition=skip_ds_if_flag_not_accepted_condition(flag),
        )
        return df

    def _combine_dfs(
            self,
            get_df_from_dataset_method: Callable[[DatasetType], pd.Series | pd.DataFrame],
            skip_ds_condition: Callable[[DatasetType], bool] = None,
            keep_first: bool = True,
    ) -> pd.Series | pd.DataFrame:
        if skip_ds_condition is None:
            skip_ds_condition = _never_skip_ds_condition
        df = combine_dfs(
            [
                get_df_from_dataset_method(ds)
                for ds in self.dataset_iterator
                if not skip_ds_condition(ds)
            ],
            keep_first=keep_first
        )
        return df


class DatasetConcatCollection(
    Generic[DatasetType, DatasetConfigType, FlagType, FlagIndexType],
    DatasetCollection[DatasetType, DatasetConfigType, FlagType, FlagIndexType]
):
    """
    Fetch method will return a concatenation of all sub-Datasets with an additional Index-level.
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
            {ds.name: ds.get_kpi_series(**kwargs) for ds in self.dataset_iterator},
            axis=1,
            names=[self.concat_level_name]
        )

    def _fetch(
            self,
            flag: FlagType,
            effective_config: DatasetConfigType,
            skip_ds_condition: Callable[[Dataset], bool] = None,
            concat_axis: int = None,
            transpose: bool = False,
            **kwargs
    ) -> pd.Series | pd.DataFrame:
        if concat_axis is None:
            concat_axis = self.default_concat_axis

        if skip_ds_condition is None:
            skip_ds_condition = skip_ds_if_flag_not_accepted_condition(flag)

        df = self._concat_dataset_dfs(
            get_df_from_dataset_method=lambda ds: ds.fetch(flag, effective_config, **kwargs),
            skip_ds_condition=skip_ds_condition,
            axis=concat_axis,
            transpose=transpose
        )
        return df

    def _concat_dataset_dfs(
            self,
            get_df_from_dataset_method: Callable[[DatasetType], pd.Series | pd.DataFrame],
            skip_ds_condition: Callable[[DatasetType], bool] = None,
            axis: int = 1,
            transpose: bool = False,
    ) -> pd.Series | pd.DataFrame:
        if skip_ds_condition is None:
            skip_ds_condition = _never_skip_ds_condition
        dfs = {
            ds.name: get_df_from_dataset_method(ds)
            for ds in self.dataset_iterator
            if not skip_ds_condition(ds)
        }

        if len(dfs):
            df0 = list(dfs.values())[0]
            if not all(len(df.axes) == len(df0.axes) for df in dfs.values()):
                raise NotImplementedError(f'Axes lengths do not match between dfs.')
            for ax in range(len(list(dfs.values())[0].axes)):
                if not all(set(df.axes[ax].names) == set(df0.axes[ax].names) for df in dfs.values()):
                    raise NotImplementedError(f'Axes names do not match between dfs.')

            # from mescal.utils.pandas_utils.standardize_indices import standardize_index
            # dfs = standardize_index(dfs, axis=axis)

        df = pd.concat(dfs, join='outer', axis=axis, names=[self.concat_level_name])

        if not self.concat_top:
            ax: pd.MultiIndex = df.axes[axis]
            df.axes[axis] = ax.reorder_levels([ax.nlevels] + list(range(ax.nlevels - 1)))

        if transpose:
            return df.transpose()
        return df


class DatasetSumCollection(
    Generic[DatasetType, DatasetConfigType, FlagType, FlagIndexType],
    DatasetCollection[DatasetType, DatasetConfigType, FlagType, FlagIndexType]
):
    def _fetch(self, flag: FlagType, effective_config: DatasetConfigType, **kwargs) -> pd.Series | pd.DataFrame:
        data: list[pd.Series | pd.DataFrame] = []
        for ds in self.dataset_iterator:
            if ds.flag_is_accepted(flag):
                data.append(ds.fetch(flag, effective_config, **kwargs))
        if not data:
            raise KeyError(f"Flag '{flag}' not recognized by any of the linked Datasets.")
        
        if all(pd_is_numeric(d) for d in data):
            import numpy as np
            return np.sum(data)

        raise NotImplementedError
