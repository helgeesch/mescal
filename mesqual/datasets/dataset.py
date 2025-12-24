from __future__ import annotations

from typing import TYPE_CHECKING, Union, Type, Iterable, Generic
from abc import ABC, abstractmethod

import pandas as pd

from mesqual.typevars import DatasetConfigType, FlagType, FlagIndexType
from mesqual.databases.database import Database
from mesqual.flag.flag_index import EmptyFlagIndex
from mesqual.utils.logging import get_logger

if TYPE_CHECKING:
    from mesqual.datasets.dataset_collection import DatasetLinkCollection
    from mesqual.kpis.kpi import KPI
    from mesqual.kpis.definitions.base import KPIDefinition
    from mesqual.kpis.collection import KPICollection

logger = get_logger(__name__)


def flag_must_be_accepted(method):
    """
    Decorator that validates flag acceptance before method execution.
    
    Ensures that only accepted flags are processed by dataset methods,
    providing clear error messages for invalid flag usage.
    
    Args:
        method: The method to decorate
        
    Returns:
        Decorated method that validates flag acceptance
        
    Raises:
        ValueError: If the flag is not accepted by the dataset
    """
    def raise_if_flag_not_accepted(self: Dataset, flag: FlagType, config: DatasetConfigType = None, **kwargs):
        if not self.flag_is_accepted(flag):
            raise ValueError(f'Flag {flag} not accepted by Dataset "{self.name}" of type {type(self)}.')
        return method(self, flag, config, **kwargs)
    return raise_if_flag_not_accepted


class _DotNotationFetcher:
    """
    Enables dot notation access for Dataset flag fetching.

    Accumulates flag parts through attribute access and converts them to a flag via
    the dataset's flag_index when executed. Supports both immediate execution through
    direct dataset attribute access and delayed execution through fetch_dotted.

    Usage:
        dataset.dotfetch().my.flag.as.string()
    """
    def __init__(self, dataset, accumulated_parts: list[str] = None):
        self._dataset = dataset
        self._accumulated_parts = accumulated_parts or []

    def __getattr__(self, part: str) -> '_DotNotationFetcher':
        return _DotNotationFetcher(self._dataset, self._accumulated_parts + [part])

    def __str__(self) -> str:
        return '.'.join(self._accumulated_parts)

    def __call__(self) -> pd.DataFrame | pd.Series:
        return self._dataset.fetch(self._dataset.flag_index.get_flag_from_string(str(self)))


class Dataset(Generic[DatasetConfigType, FlagType, FlagIndexType], ABC):
    """
    Abstract base class for all datasets in the MESQUAL framework.
    
    The Dataset class provides the fundamental interface for data access and manipulation
    in MESQUAL. It implements the core principle "Everything is a Dataset" where individual
    scenarios, scenarios merged from multiple simulation runs or data sources,
    collections of scenarios, and scenario comparisons all share the same unified interface.
    
    Key Features:
        - Unified `.fetch(flag)` interface for data access
        - Attribute management for scenario metadata
        - KPI calculation integrations
        - Database caching support
        - Dot notation fetching via `dotfetch()`
        - Type-safe generic implementation
    
    Type Parameters:
        DatasetConfigType: Configuration class for dataset behavior
        FlagType: Type used for data flag identification (typically str)
        FlagIndexType: Flag index implementation for flag mapping
        
    Attributes:
        name (str): Human-readable identifier for the dataset
        kpi_collection (KPICollection): Collection of KPIs associated with this dataset

    Example:

        >>> # Basic usage pattern
        >>> data = dataset.fetch('buses_t.marginal_price')
        >>> flags = dataset.accepted_flags
        >>> if dataset.flag_is_accepted('generators_t.p'):
        ...     gen_data = dataset.fetch('generators_t.p')
    """
    
    def __init__(
            self,
            name: str = None,
            parent_dataset: Dataset = None,
            flag_index: FlagIndexType = None,
            attributes: dict = None,
            database: Database = None,
            config: DatasetConfigType = None
    ):
        """
        Initialize a new Dataset instance.
        
        Args:
            name: Human-readable identifier. If None, auto-generates from class name
            parent_dataset: Optional parent dataset for hierarchical relationships
            flag_index: Index for mapping and validating data flags
            attributes: Dictionary of metadata attributes for the dataset
            database: Optional database for caching expensive computations
            config: Configuration object controlling dataset behavior
        """
        self.name = name or f'{self.__class__.__name__}_{str(id(self))}'
        self._flag_index = flag_index or EmptyFlagIndex()
        self._parent_dataset = parent_dataset
        self._attributes: dict = attributes or dict()
        self._database = database
        self._config = config

        from mesqual.kpis.collection import KPICollection
        self.kpi_collection: KPICollection = KPICollection()

    def dotfetch(self) -> _DotNotationFetcher:
        return _DotNotationFetcher(self)

    @property
    def flag_index(self) -> FlagIndexType:
        if isinstance(self._flag_index, EmptyFlagIndex):
            logger.info(
                f"Dataset {self.name}: "
                "You're trying to use functionality of the FlagIndex but didn't define one. "
                "The current FlagIndex in use is empty. "
                "Make sure to set a flag_index in case you want to use full functionality of the flag_index."
            )
        return self._flag_index

    @property
    def database(self) -> Database | None:
        return self._database

    def add_kpis_from_definitions(self, kpi_definitions: KPIDefinition | list[KPIDefinition]):
        from mesqual.kpis.definitions.base import KPIDefinition
        if isinstance(kpi_definitions, KPIDefinition):
            kpis = kpi_definitions.generate_kpis(self)
            self.add_kpis(kpis)
        else:
            for kpi_def in kpi_definitions:
                kpis = kpi_def.generate_kpis(self)
                self.add_kpis(kpis)

    def add_kpis(self, kpis: Iterable[KPI]):
        """
        Add multiple KPIs to this dataset's KPI collection.
        
        Args:
            kpis: Iterable of KPI instances, factories, or classes to add
        """
        duplicates = []
        for kpi in kpis:
            if kpi in self.kpi_collection:
                duplicates.append(kpi)
            else:
                self.add_kpi(kpi)
        if duplicates:
            _num_duplicates = len(duplicates)
            logger.warning(f'{_num_duplicates} duplicates found and not added again or overwritten in {self.name}. ({duplicates[:3]}...)')

    def add_kpi(self, kpi: KPI):
        """
        Add a single KPI to this dataset's KPI collection.
        
        Args:
            kpi: KPI instance, factory, or class to add
        """
        self.kpi_collection.add(kpi)

    def clear_kpi_collection(self):
        """Clear the KPI collection."""
        from mesqual.kpis.collection import KPICollection
        self.kpi_collection = KPICollection()

    @property
    def attributes(self) -> dict:
        return self._attributes

    def get_attributes_series(self) -> pd.Series:
        att_series = pd.Series(self.attributes, name=self.name)
        return att_series

    def set_attributes(self, **kwargs):
        for key, value in kwargs.items():
            if not isinstance(key, str):
                raise TypeError(f'Attribute keys must be of type str. Your key {key} is of type {type(key)}.')
            if not isinstance(value, (bool, int, float, str)):
                raise TypeError(
                    f'Attribute values must be of type (bool, int, flaot, str). '
                    f'Your value for {key} ({value}) is of type {type(value)}.'
                )
            self._attributes[key] = value

    @property
    def parent_dataset(self) -> 'DatasetLinkCollection':
        if self._parent_dataset is None:
            raise RuntimeError(f"Parent dataset called without / before assignment.")
        return self._parent_dataset

    @parent_dataset.setter
    def parent_dataset(self, parent_dataset: 'DatasetLinkCollection'):
        from mesqual.datasets.dataset_collection import DatasetLinkCollection
        if not isinstance(parent_dataset, DatasetLinkCollection):
            raise TypeError(f"Parent parent_dataset must be of type {DatasetLinkCollection.__name__}")
        self._parent_dataset = parent_dataset

    @property
    @abstractmethod
    def accepted_flags(self) -> set[FlagType]:
        """
        Set of all flags accepted by this dataset.
        
        This abstract property must be implemented by all concrete dataset classes
        to define which data flags can be fetched from the dataset.
        
        Returns:
            Set of flags that can be used with the fetch() method
            
        Example:

            >>> print(dataset.accepted_flags)
                {'buses', 'buses_t.marginal_price', 'generators', 'generators_t.p', ...}
        """
        return set()

    def get_accepted_flags_containing_x(self, x: str, match_case: bool = False) -> set[FlagType]:
        """
        Find all accepted flags containing a specific substring.
        
        Useful for discovering related data flags or filtering flags by category.
        
        Args:
            x: Substring to search for in flag names
            match_case: If True, performs case-sensitive search. Default is False.
            
        Returns:
            Set of accepted flags containing the substring
            
        Example:

            >>> ds = PyPSADataset()
            >>> ds.get_accepted_flags_containing_x('generators')
                {'generators', 'generators_t.p', 'generators_t.efficiency', ...}
            >>> ds.get_accepted_flags_containing_x('BUSES', match_case=True)
                set()  # Empty because case doesn't match
        """
        if match_case:
            return {f for f in self.accepted_flags if x in str(f)}
        x_lower = x.lower()
        return {f for f in self.accepted_flags if x_lower in str(f).lower()}

    def flag_is_accepted(self, flag: FlagType) -> bool:
        """
        Boolean check whether a flag is accepted by the Dataset.

        This method can be optionally overridden in any child-class
        in case you want to follow logic instead of the explicit set of accepted_flags.
        """
        return flag in self.accepted_flags

    @flag_must_be_accepted
    def required_flags_for_flag(self, flag: FlagType) -> set[FlagType]:
        return self._required_flags_for_flag(flag)

    @abstractmethod
    def _required_flags_for_flag(self, flag: FlagType) -> set[FlagType]:
        return set()

    @flag_must_be_accepted
    def fetch(self, flag: FlagType, config: dict | DatasetConfigType = None, **kwargs) -> pd.Series | pd.DataFrame:
        """
        Fetch data associated with a specific flag.

        This is the primary method for data access in MESQUAL datasets. It provides
        a unified interface for retrieving data regardless of the underlying source
        or dataset type. The method includes automatic caching, post-processing,
        and configuration management.

        Configuration Override Behavior
        -------------------------------
        The ``config`` parameter allows fetch-time overrides of dataset behavior.
        These overrides are merged with the dataset's effective configuration
        (which combines class-level and instance-level settings). Only non-None
        values in the override will replace the existing settings.

        The configuration resolution hierarchy (later overrides earlier):
            1. Base config defaults
            2. Class config (via DatasetConfigManager)
            3. Instance config (passed to Dataset.__init__)
            4. **Fetch-time config (this parameter)**

        Args:
            flag: Data identifier flag (must be in accepted_flags)
            config: Optional configuration to override dataset defaults.
                Can be either:

                - **dict**: Quick way to override specific settings. Keys must
                  match config attribute names (e.g., ``use_database``,
                  ``auto_sort_datetime_index``). Platform-specific options
                  are also supported if the dataset uses an extended config.

                - **DatasetConfig instance**: Full config object for type safety.
                  Must be compatible with the dataset's config type.

            **kwargs: Additional keyword arguments passed to the underlying
                data fetching implementation

        Returns:
            DataFrame or Series containing the requested data

        Raises:
            ValueError: If the flag is not accepted by this dataset

        Examples:
            Basic usage::

                >>> prices = dataset.fetch('buses_t.marginal_price')

            Override base config options with a dict::

                >>> # Skip database cache for this fetch
                >>> prices = dataset.fetch(
                ...     'buses_t.marginal_price',
                ...     config=dict(use_database=False)
                ... )
                >>>
                >>> # Disable datetime sorting
                >>> prices = dataset.fetch(
                ...     'generators_t.p',
                ...     config=dict(auto_sort_datetime_index=False)
                ... )

            Override platform-specific options::

                >>> # Platform configs may have additional options
                >>> # e.g., a config with timestamp conversion setting
                >>> data = dataset.fetch(
                ...     'some_flag',
                ...     config=dict(convert_period_enum_to_datetime_index=False)
                ... )

            Override study-specific options::

                >>> # Study-specific configs can add custom behavior
                >>> # e.g., toggle custom data corrections
                >>> data = dataset.fetch(
                ...     'some_flag',
                ...     config=dict(apply_custom_correction=False)
                ... )

            Using a config object::

                >>> from mesqual.datasets import DatasetConfig
                >>> custom_config = DatasetConfig(
                ...     use_database=False,
                ...     auto_sort_datetime_index=False
                ... )
                >>> prices = dataset.fetch('buses_t.marginal_price', config=custom_config)

        See Also:
            :class:`~mesqual.datasets.dataset_config.DatasetConfig`: Base configuration class
            :class:`~mesqual.datasets.dataset_config.DatasetConfigManager`: Class-level config registry
        """
        effective_config = self._prepare_config(config)
        use_database = self._database is not None and effective_config.use_database

        if use_database:
            if self._database.key_is_up_to_date(self, flag, config=effective_config, **kwargs):
                return self._database.get(self, flag, config=effective_config, **kwargs)

        raw_data = self._fetch(flag, effective_config, **kwargs)
        processed_data = self._post_process_data(raw_data, flag, effective_config)

        if use_database:
            self._database.set(self, flag, config=effective_config, value=processed_data, **kwargs)

        return processed_data.copy()

    def _post_process_data(
            self,
            data: pd.Series | pd.DataFrame,
            flag: FlagType,
            config: DatasetConfigType
    ) -> pd.Series | pd.DataFrame:
        if config.remove_duplicate_indices and any(data.index.duplicated()):
            logger.info(
                f'For some reason your data-set {self.name} returns an object with duplicate indices for flag {flag}.\n'
                f'We manually remove duplicate indices. Please make sure your data importer / converter is set up '
                f'appropriately and that your raw data does not contain duplicate indices. \n'
                f'We will keep the first element of every duplicated index.'
            )
            data = data.loc[~data.index.duplicated()]
        if config.auto_sort_datetime_index and isinstance(data.index, pd.DatetimeIndex):
            data = data.sort_index()
        return data

    def _prepare_config(self, config: dict | DatasetConfigType = None) -> DatasetConfigType:
        """
        Prepare the effective configuration for a fetch operation.

        Resolves the final configuration by merging the provided config override
        with the dataset's instance config (which already includes class-level
        and base defaults via DatasetConfigManager).

        Args:
            config: Optional override configuration. Can be:
                - None: Use instance config as-is
                - dict: Create temp config from dict and merge
                - DatasetConfig: Merge directly with instance config

        Returns:
            The fully resolved configuration for the fetch operation.

        Raises:
            TypeError: If config is neither None, dict, nor DatasetConfig.
        """
        if config is None:
            return self.instance_config

        if isinstance(config, dict):
            temp_config = self.get_config_type()()
            temp_config.__dict__.update(config)
            return self.instance_config.merge(temp_config)

        from mesqual.datasets.dataset_config import DatasetConfig
        if isinstance(config, DatasetConfig):
            return self.instance_config.merge(config)

        raise TypeError(f"Config must be dict or {DatasetConfig.__name__}, got {type(config)}")

    @abstractmethod
    def _fetch(self, flag: FlagType, effective_config: DatasetConfigType, **kwargs) -> pd.Series | pd.DataFrame:
        return pd.DataFrame()

    def fetch_multiple_flags_and_concat(
            self,
            flags: Iterable[FlagType],
            concat_axis: int = 1,
            concat_level_name: str = 'variable',
            concat_level_at_top: bool = True,
            config: dict | DatasetConfigType = None,
            **kwargs
    ) -> Union[pd.Series, pd.DataFrame]:
        dfs = {
            str(flag): self.fetch(flag, config, **kwargs)
            for flag in flags
        }
        df = pd.concat(
            dfs,
            axis=concat_axis,
            names=[concat_level_name],
        )
        if not concat_level_at_top:
            ax = df.axes[concat_axis]
            ax = ax.reorder_levels(list(range(1, ax.nlevels)) + [0])
            df.axes[concat_axis] = ax
        return df

    def fetch_filter_groupby_agg(
            self,
            flag: FlagType,
            model_filter_query: str = None,
            prop_groupby: str | list[str] = None,
            prop_groupby_agg: str = None,
            config: dict | DatasetConfigType = None,
            **kwargs
    ) -> pd.Series | pd.DataFrame:
        model_flag = self.flag_index.get_linked_model_flag(flag)
        if not model_flag:
            raise RuntimeError(f'FlagIndex could not successfully map flag {flag} to a model flag.')

        from mesqual.utils import pandas_utils

        data = self.fetch(flag, config, **kwargs)
        model_df = self.fetch(model_flag, config, **kwargs)

        if model_filter_query:
            data = pandas_utils.filter_by_model_query(data, model_df, query=model_filter_query)

        if prop_groupby:
            if isinstance(prop_groupby, str):
                prop_groupby = [prop_groupby]
            data = pandas_utils.prepend_model_prop_levels(data, model_df, *prop_groupby)
            data = data.groupby(prop_groupby)
            if prop_groupby_agg:
                data = data.agg(prop_groupby_agg)
        elif prop_groupby_agg:
            logger.warning(
                f"You provided a prop_groupby_agg operation, but didn't provide prop_groupby. "
                f"No aggregation performed."
            )
        return data

    @classmethod
    def get_flag_type(cls) -> Type[FlagType]:
        from mesqual.flag.flag import FlagTypeProtocol
        return FlagTypeProtocol

    @classmethod
    def get_flag_index_type(cls) -> Type[FlagIndexType]:
        from mesqual.flag.flag_index import FlagIndex
        return FlagIndex

    @classmethod
    def get_config_type(cls) -> Type[DatasetConfigType]:
        from mesqual.datasets.dataset_config import DatasetConfig
        return DatasetConfig

    @property
    def instance_config(self) -> DatasetConfigType:
        from mesqual.datasets.dataset_config import DatasetConfigManager
        return DatasetConfigManager.get_effective_config(self.__class__, self._config)

    def set_instance_config(self, config: DatasetConfigType) -> None:
        self._config = config

    def set_instance_config_kwargs(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self._config, key, value)

    @classmethod
    def set_class_config(cls, config: DatasetConfigType) -> None:
        from mesqual.datasets.dataset_config import DatasetConfigManager
        DatasetConfigManager.set_class_config(cls, config)

    def __str__(self) -> str:
        return self.name

    def __hash__(self):
        return hash((self.name, self._config))
