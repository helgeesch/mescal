from __future__ import annotations

from typing import Generic, Iterable, TYPE_CHECKING, Iterator
from abc import ABC, abstractmethod

import pandas as pd

from mesqual.datasets.dataset import Dataset
from mesqual.flag.flag_index import FlagIndex
from mesqual.utils.pandas_utils.is_numeric import pd_is_numeric
from mesqual.utils.logging import get_logger
from mesqual.utils.set_aggregations import nested_union
from mesqual.utils.intersect_dicts import get_intersection_of_dicts
from mesqual.typevars import DatasetType, DatasetConfigType, FlagType, FlagIndexType

if TYPE_CHECKING:
    from mesqual.kpis.kpi import KPI
    from mesqual.kpis.definitions.base import KPIDefinition
    from mesqual.kpis.collection import KPICollection
    from mesqual.databases.database import Database

logger = get_logger(__name__)


class DatasetCollection(
    Generic[DatasetType, DatasetConfigType, FlagType, FlagIndexType],
    Dataset[DatasetConfigType, FlagType, FlagIndexType],
    ABC
):
    """
    Abstract base class for collections of datasets.
    
    DatasetCollection extends the Dataset interface to handle multiple child datasets
    while maintaining the same unified API. This enables complex hierarchical structures
    where collections themselves can be treated as datasets.
    
    Key Features:
        - Inherits all Dataset functionality
        - Manages collections of child datasets
        - Provides iteration and access methods
        - Aggregates accepted flags from all children
        - Supports KPI operations across all sub-datasets
        
    Type Parameters:
        DatasetType: Type of datasets that can be collected
        DatasetConfigType: Configuration class for dataset behavior
        FlagType: Type used for data flag identification
        FlagIndexType: Flag index implementation for flag mapping
        
    Attributes:
        datasets (list[DatasetType]): List of child datasets in this collection
        
    Note:
        This class follows the "Everything is a Dataset" principle, allowing
        collections to be used anywhere a Dataset is expected.
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
    def dataset_iterator(self) -> Iterator[DatasetType]:
        for ds in self.datasets:
            yield ds

    @property
    def flag_index(self) -> FlagIndex:
        from mesqual.flag.flag_index import EmptyFlagIndex
        if (self._flag_index is None) or isinstance(self._flag_index, EmptyFlagIndex):
            from mesqual.utils.check_all_same import all_same_object
            if all_same_object(ds.flag_index for ds in self.datasets) and len(self.datasets):
                return self.get_dataset().flag_index
        return self._flag_index

    @property
    def attributes(self) -> dict:
        child_dataset_atts = [ds.attributes for ds in self.datasets]
        attributes_that_all_childs_have_in_common = get_intersection_of_dicts(child_dataset_atts)
        return {**attributes_that_all_childs_have_in_common, **self._attributes.copy()}

    def get_merged_kpi_collection(self, deep: bool = True) -> KPICollection:
        """
        Merge KPI collections from all child datasets.

        This method collects KPIs from all child datasets' kpi_collection
        properties and returns a unified collection. Optionally recurses into
        nested DatasetCollections.

        Args:
            deep: If True, recursively merge from nested DatasetCollections

        Returns:
            KPICollection containing all KPIs from all child datasets

        Example:

            >>> # Create KPIs for all scenarios
            >>> study.scen: DatasetConcatCollection
            >>> study.scen.add_kpis_from_definitions_to_all_child_datasets(kpi_defs)
            >>>
            >>> # Get merged collection across all scenarios
            >>> all_kpis = study.scen.get_merged_kpi_collection()
            >>>
            >>> # Filter and export
            >>> mean_prices = all_kpis.filter_by(aggregation=Aggregations.Mean)
            >>> df = mean_prices.to_dataframe(unit_handling='auto_convert')
        """
        from mesqual.kpis.collection import KPICollection
        merged = KPICollection()

        for ds in self.datasets:
            # Add KPIs from this dataset
            merged.extend(ds.kpi_collection._kpis)

            # Recursively add from nested collections
            if deep and isinstance(ds, DatasetCollection):
                nested_merged = ds.get_merged_kpi_collection(deep=deep)
                merged.extend(nested_merged._kpis)

        return merged

    def add_kpis_from_definitions_to_all_child_datasets(self, kpi_definitions: KPIDefinition | list[KPIDefinition]):
        for ds in self.dataset_iterator:
            ds.add_kpis_from_definitions(kpi_definitions)

    def clear_kpi_collection_for_all_child_datasets(self, deep: bool = True):
        for ds in self.datasets:
            ds.clear_kpi_collection()
            if deep and isinstance(ds, DatasetCollection):
                ds.clear_kpi_collection_for_all_child_datasets(deep=deep)

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

    def fetch_merged(
            self,
            flag: FlagType,
            config: dict | DatasetConfigType = None,
            keep_first: bool = True,
            **kwargs
    ) -> pd.Series | pd.DataFrame:
        """Fetch method that merges dataframes from all child datasets, similar to DatasetMergeCollection."""
        temp_merge_collection = self.get_merged_dataset_collection(keep_first)
        return temp_merge_collection.fetch(flag, config, **kwargs)

    def get_merged_dataset_collection(self, keep_first: bool = True) -> 'DatasetMergeCollection':
        return DatasetMergeCollection(
            datasets=self.datasets,
            name=f"{self.name} merged",
            keep_first=keep_first
        )


class DatasetLinkCollection(
    Generic[DatasetType, DatasetConfigType, FlagType, FlagIndexType],
    DatasetCollection[DatasetType, DatasetConfigType, FlagType, FlagIndexType]
):
    """
    Links multiple datasets to provide unified data access with automatic routing.
    
    DatasetLinkCollection acts as a unified interface to multiple child datasets,
    automatically routing data requests to the appropriate child dataset that 
    accepts the requested flag. This is the foundation for platform datasets
    that combine multiple data interpreters.
    
    Key Features:
        - Automatic flag routing to appropriate child dataset
        - Bidirectional parent-child relationships
        - First-match-wins routing strategy
        - Overlap detection and warnings
        - Maintains all Dataset interface compatibility
        
    Routing Logic:
        When fetch() is called, iterates through child datasets in order and
        returns data from the first dataset that accepts the flag.
        
    Example:

        >>> # Platform dataset with multiple interpreters
        >>> link_collection = DatasetLinkCollection([
        ...     ModelInterpreter(network),
        ...     TimeSeriesInterpreter(network),
        ...     ObjectiveInterpreter(network)
        ... ])
        >>> # Automatically routes to appropriate interpreter
        >>> buses = link_collection.fetch('buses')  # -> ModelInterpreter
        >>> prices = link_collection.fetch('buses_t.marginal_price')  # -> TimeSeriesInterpreter
        
    Warning:
        If multiple child datasets accept the same flag, only the first one
        will be used. The constructor logs warnings for such overlaps.
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

    def get_dataset_by_type(self, ds_type: type[Dataset]) -> DatasetType:
        """Returns instance of child dataset that matches the ds_type."""
        for ds in self.datasets:
            if isinstance(ds, ds_type):
                return ds
        raise KeyError(f'No Dataset of type {ds_type.__name__} found in {self.name}.')


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

        from mesqual.utils.pandas_utils.combine_df import combine_dfs
        df = combine_dfs(data_frames, keep_first=self.keep_first)
        return df


class DatasetConcatCollection(
    Generic[DatasetType, DatasetConfigType, FlagType, FlagIndexType],
    DatasetCollection[DatasetType, DatasetConfigType, FlagType, FlagIndexType]
):
    """
    Concatenates data from multiple datasets with MultiIndex structure.
    
    DatasetConcatCollection is fundamental to MESQUAL's multi-scenario analysis
    capabilities. It fetches the same flag from multiple child datasets and
    concatenates the results into a single DataFrame/Series with an additional
    index level identifying the source dataset.
    
    Key Features:
        - Automatic MultiIndex creation with dataset names
        - Configurable concatenation axis and level positioning  
        - Preserves all dimensional relationships
        - Supports scenario and comparison collections
        - Enables unified analysis across multiple datasets
        
    MultiIndex Structure:
        The resulting data structure includes an additional index level
        (typically named 'dataset') that identifies the source dataset
        for each data point.
        
    Example:

        >>> # Collection of scenario datasets
        >>> scenarios = DatasetConcatCollection([
        ...     PyPSADataset(base_network, name='base'),
        ...     PyPSADataset(high_res_network, name='high_res'),
        ...     PyPSADataset(low_gas_network, name='low_gas')
        ... ])
        >>> 
        >>> # Fetch creates MultiIndex DataFrame
        >>> prices = scenarios.fetch('buses_t.marginal_price')
        >>> print(prices.columns.names)
            ['dataset', 'Bus']  # Original Bus index + dataset level
        >>> 
        >>> # Access specific scenario data
        >>> base_prices = prices['base']
        >>> 
        >>> # Analyze across scenarios
        >>> mean_prices = prices.mean()  # Mean across all scenarios
    """
    DEFAULT_CONCAT_LEVEL_NAME = 'dataset'
    DEFAULT_ATT_LEVEL_NAME = 'attribute'

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
        self.concat_level_name = concat_level_name or self.DEFAULT_CONCAT_LEVEL_NAME

    def get_attributes_concat_df(self) -> pd.DataFrame:
        if all(isinstance(ds, DatasetConcatCollection) for ds in self.datasets):
            use_att_df_instead_of_series = True
        else:
            use_att_df_instead_of_series = False

        atts_per_dataset = dict()
        for ds in self.datasets:
            atts = ds.get_attributes_concat_df().T if use_att_df_instead_of_series else ds.get_attributes_series()
            atts_per_dataset[ds.name] = atts

        return pd.concat(
            atts_per_dataset,
            axis=1,
            names=[self.concat_level_name]
        ).rename_axis(self.DEFAULT_ATT_LEVEL_NAME).T

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
            raise KeyError(f"Flag '{flag}' not recognized by any of the datasets in {type(self)} {self.name}.")

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
            raise KeyError(f"Flag '{flag}' not recognized by any of the linked Datasets in {type(self)} {self.name}.")
        
        if all(pd_is_numeric(d) for d in data):
            import numpy as np
            return np.sum(data)

        raise NotImplementedError
