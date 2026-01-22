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

    @flag_must_be_accepted
    def fetch(self, flag: FlagType, config: dict | DatasetConfigType = None, **kwargs) -> pd.Series | pd.DataFrame:
        """
        Fetch data associated with a specific flag.

        This is the primary method for data access in MESQUAL datasets. It provides
        a unified interface for retrieving data regardless of the underlying source
        or dataset type. The method includes automatic caching, post-processing,
        and configuration management.

        Configuration Override Behavior:

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

    def dotfetch(self) -> _DotNotationFetcher:
        """
        Create a dot notation fetcher for intuitive flag access.

        Returns a helper object that allows accessing nested data flags using
        Python attribute syntax instead of string-based flags. The fetcher
        accumulates attribute accesses and converts them to the appropriate
        flag when called.

        Returns:
            _DotNotationFetcher: Helper object enabling chained attribute access

        Example:
            Using dot notation instead of string flags::

                >>> # Traditional string-based fetch
                >>> prices = dataset.fetch('buses_t.marginal_price')

                >>> # Equivalent dot notation fetch
                >>> prices = dataset.dotfetch().buses_t.marginal_price()

                >>> # Multi-level flag access
                >>> gen_power = dataset.dotfetch().generators_t.p()
        """
        return _DotNotationFetcher(self)

    @property
    def flag_index(self) -> FlagIndexType:
        """
        Access the flag index for this dataset.

        The flag index provides flag mapping, validation, and metadata lookup
        capabilities. It enables features like dot notation fetching, flag-to-model
        mapping, and flag categorization.

        If no flag index was configured, returns an EmptyFlagIndex and logs
        an informational message when accessed.

        Returns:
            FlagIndexType: The configured flag index or EmptyFlagIndex if none set

        Note:
            For full flag index functionality (model mapping, flag categorization),
            ensure a proper flag index is set during dataset initialization.

        See Also:
            - [Flag System](../flag.md) - Flag index implementations and usage
        """
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
        """
        Access the caching database for this dataset.

        The database provides persistent caching for expensive fetch operations.
        When configured, the fetch() method automatically checks the database
        before computing data and stores results for future access.

        Returns:
            Database | None: The configured database instance, or None if caching
                is not enabled for this dataset

        See Also:
            - Database configuration and caching behavior
            - Uses database for automatic caching when available (see `fetch()` method)
        """
        return self._database

    def add_kpis_from_definitions(self, kpi_definitions: KPIDefinition | list[KPIDefinition]):
        """
        Generate and add KPIs from one or more KPI definitions.

        KPI definitions are templates that generate concrete KPI instances
        based on the dataset's structure. This method processes definitions
        and adds the resulting KPIs to the dataset's KPI collection.

        Args:
            kpi_definitions: Single KPIDefinition or list of definitions.
                Each definition's generate_kpis() method is called with
                this dataset to produce KPI instances.

        Example:
            Adding KPIs from definitions::

                >>> from mesqual.kpis.definitions import TotalGenerationKPIDefinition
                >>> dataset.add_kpis_from_definitions(TotalGenerationKPIDefinition())

                >>> # Add multiple definitions at once
                >>> definitions = [
                ...     TotalGenerationKPIDefinition(),
                ...     MarginalPriceKPIDefinition(),
                ... ]
                >>> dataset.add_kpis_from_definitions(definitions)

        See Also:
            - `add_kpi()` - Add a single KPI directly
            - `add_kpis()` - Add multiple KPI instances
            - [KPI Definitions](../kpis/definitions/base.md) - Base KPI definition class
        """
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
        """
        Access the metadata attributes dictionary for this dataset.

        Attributes store scenario-level metadata such as configuration parameters,
        simulation settings, or descriptive labels. These are useful for filtering,
        grouping, and annotating datasets in collections.

        Returns:
            dict: Dictionary of attribute key-value pairs. Keys are strings,
                values are primitive types (bool, int, float, str).

        Example:
            Accessing and using attributes::

                >>> dataset.attributes
                {'year': 2030, 'scenario_type': 'high_renewable', 'carbon_price': 50.0}

                >>> # Filter datasets in a collection by attribute
                >>> high_re_scenarios = [d for d in collection if d.attributes.get('scenario_type') == 'high_renewable']

        See Also:
            - `set_attributes()` - Set attribute values
            - `get_attributes_series()` - Convert attributes to pandas Series
        """
        return self._attributes

    def get_attributes_series(self) -> pd.Series:
        """
        Convert dataset attributes to a pandas Series.

        Creates a Series with attribute names as the index and attribute
        values as data. The Series name is set to the dataset name, making
        it suitable for concatenation with other datasets' attribute series.

        Returns:
            pd.Series: Series containing attribute values, indexed by attribute
                names, with the dataset name as the Series name

        Example:
            Converting attributes and combining across datasets::

                >>> dataset.set_attributes(year=2030, carbon_price=50.0)
                >>> series = dataset.get_attributes_series()
                >>> series
                year            2030
                carbon_price    50.0
                Name: Scenario_A, dtype: object

                >>> # Combine attributes from multiple datasets
                >>> attr_df = pd.concat([d.get_attributes_series() for d in collection], axis=1).T

        See Also:
            - `attributes` - Access raw attributes dictionary
            - `set_attributes()` - Set attribute values
        """
        att_series = pd.Series(self.attributes, name=self.name)
        return att_series

    def set_attributes(self, **kwargs):
        """
        Set one or more metadata attributes on this dataset.

        Attributes are key-value pairs that store scenario metadata. They must
        use string keys and primitive values (bool, int, float, str) to ensure
        serializability and consistent comparison behavior.

        Args:
            **kwargs: Attribute key-value pairs to set. Keys must be strings,
                values must be bool, int, float, or str.

        Raises:
            TypeError: If any key is not a string
            TypeError: If any value is not bool, int, float, or str

        Example:
            Setting scenario metadata::

                >>> dataset.set_attributes(
                ...     year=2030,
                ...     scenario_type='high_renewable',
                ...     carbon_price=50.0,
                ...     includes_nuclear=True
                ... )

                >>> # Access the attributes
                >>> dataset.attributes['year']
                2030

        See Also:
            - `attributes` - Access attributes dictionary
            - `get_attributes_series()` - Convert to pandas Series
        """
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
        """
        Access the parent collection linking this interpreter to sibling interpreters.

        The parent_dataset provides the bridge between specialized flag interpreters
        within a single platform dataset or study. It is NOT used to link scenarios
        together, but rather to enable modular interpreter architectures where each
        interpreter handles a specific subset of flags and can access flags from
        sibling interpreters through the shared parent.

        Architecture Pattern:
            A typical platform dataset (e.g., PyPSADataset, PlexosDataset) is
            implemented as a DatasetLinkCollection containing multiple specialized
            interpreters:

            - **ModelInterpreter**: Provides static model data (e.g., 'generators', 'buses')
            - **TimeSeriesInterpreter**: Provides time-series data (e.g., 'generators_t.p')
            - **ObjectiveInterpreter**: Provides objective function values
            - **Custom Interpreters**: Study-specific derived or corrected variables

            Each interpreter is a child dataset within the parent DatasetLinkCollection.
            Through parent_dataset, any interpreter can fetch flags from siblings without
            needing direct references or circular dependencies.

        Why This Pattern:
            - **Separation of Concerns**: Each interpreter focuses on one data type
            - **Modularity**: Add/remove/replace interpreters independently
            - **Dependency Resolution**: Interpreters can depend on each other's flags
            - **Study Customization**: Override or extend specific interpreters per study
            - **Maintainability**: Changes to one interpreter don't affect others

        Returns:
            DatasetLinkCollection: The parent collection that orchestrates flag
                routing between this interpreter and its siblings

        Raises:
            RuntimeError: If accessed before the parent has been assigned (typically
                happens if an interpreter is used standalone instead of within a
                DatasetLinkCollection)

        Example:
            Custom interpreter combining flags from sibling interpreters:

                >>> # Study-specific interpreter for renewable generation per bidding zone
                >>> class RESGenerationPerBZInterpreter(PlatformBaseInterpreterDataset):
                ...     @property
                ...     def accepted_flags(self):
                ...         return {'generators_t.res_generation_per_bz'}
                ...
                ...     def _fetch(self, flag, config, **kwargs):
                ...         # Fetch time series from TimeSeriesInterpreter sibling
                ...         generation = self.parent_dataset.fetch('generators_t.p')
                ...
                ...         # Fetch model data from ModelInterpreter sibling
                ...         gen_model = self.parent_dataset.fetch('generators.model')
                ...
                ...         # Filter to RES generators and aggregate by bidding zone
                ...         res_gens = gen_model[gen_model['carrier'].isin(['solar', 'wind'])]
                ...         res_generation = generation[res_gens.index]
                ...         return res_generation.groupby(gen_model['bidding_zone'], axis=1).sum()

            Accessing specific sibling interpreter by type:

                >>> class SomeCustomPTDFMatrixFormat(PlexosImporterBase):
                ...     def _fetch(self, flag, config, **kwargs):
                ...         # Get specific sibling interpreter
                ...         ptdf_ds = self.parent_dataset.get_dataset_by_type(
                ...             PlexosPTDFInterpreter
                ...         )
                ...
                ...         # Or fetch through parent (automatically routes to correct sibling)
                ...         headers = self.parent_dataset.fetch('PTDF.Headers')
                ...         factors = self.parent_dataset.fetch('PTDF.Factors')
                ...
                ...         # Process and return derived flag
                ...         return self._custom_ptdf_process(headers, factors)

            Study-specific correction of platform variables:

                >>> class LineFlows(MyStudyVariables):
                ...     '''Replaces specific flows with external data.'''
                ...
                ...     def _fetch(self, flag, config, **kwargs):
                ...         # Get reference dataset (sibling interpreter) for this flag
                ...         reference_ds = self._get_reference_dataset_for_flag(flag)
                ...
                ...         # Fetch original data from sibling
                ...         df = reference_ds.fetch(flag, config, **kwargs)
                ...
                ...         # Apply study-specific corrections
                ...         if self.parent_dataset.attributes['manual_line_flow_correction']:
                ...             df = self._apply_historical_corrections(df)
                ...
                ...         return df

        See Also:
            - `DatasetLinkCollection` - Parent collection class that orchestrates routing
            - `get_dataset_by_type()` - Method to access specific sibling by type
        """
        if self._parent_dataset is None:
            raise RuntimeError(f"Parent dataset called without / before assignment.")
        return self._parent_dataset

    @parent_dataset.setter
    def parent_dataset(self, parent_dataset: 'DatasetLinkCollection'):
        """
        Set the parent collection for this dataset.

        Args:
            parent_dataset: The DatasetLinkCollection that will contain this dataset

        Raises:
            TypeError: If parent_dataset is not a DatasetLinkCollection instance
        """
        from mesqual.datasets.dataset_collection import DatasetLinkCollection
        if not isinstance(parent_dataset, DatasetLinkCollection):
            raise TypeError(f"Parent parent_dataset must be of type {DatasetLinkCollection.__name__}")
        self._parent_dataset = parent_dataset

    @flag_must_be_accepted
    def required_flags_for_flag(self, flag: FlagType) -> set[FlagType]:
        """
        Get the set of flags required to compute a given flag.

        For derived or computed flags, this method returns the set of source
        flags that must be available to produce the requested data. This is
        useful for understanding data dependencies and ensuring prerequisite
        data exists.

        Args:
            flag: The flag to check requirements for. Must be in accepted_flags.

        Returns:
            set[FlagType]: Set of flags that are required to compute the given flag.
                Returns an empty set if the flag has no dependencies.

        Raises:
            ValueError: If the flag is not accepted by this dataset

        Example:
            Checking data dependencies::

                >>> # A derived flag might depend on multiple source flags
                >>> deps = dataset.required_flags_for_flag('total_generation')
                >>> deps
                {'generators_t.p', 'generators'}

        See Also:
            - `_required_flags_for_flag()` - Abstract method to implement
            - `flag_is_accepted()` - Check if a flag is valid
        """
        return self._required_flags_for_flag(flag)

    @abstractmethod
    def _required_flags_for_flag(self, flag: FlagType) -> set[FlagType]:
        """
        Abstract method to define flag dependencies.

        Subclasses must implement this method to specify which flags are
        required to compute a given flag. This enables dependency tracking
        and validation of data availability.

        Args:
            flag: The flag to get requirements for

        Returns:
            set[FlagType]: Set of prerequisite flags. Return empty set for
                flags with no dependencies.

        Note:
            This is a protected method called by required_flags_for_flag().
            The public method handles flag validation before calling this.
        """
        return set()

    def _post_process_data(
            self,
            data: pd.Series | pd.DataFrame,
            flag: FlagType,
            config: DatasetConfigType
    ) -> pd.Series | pd.DataFrame:
        """
        Apply standard post-processing to fetched data.

        Performs configuration-driven data cleaning and normalization after
        the raw data is fetched. This includes removing duplicate indices
        and sorting datetime indices.

        Args:
            data: Raw data from _fetch()
            flag: The flag that was fetched (for logging)
            config: Effective configuration controlling post-processing behavior

        Returns:
            Post-processed data with duplicates removed and/or sorted as configured

        Note:
            This method is called automatically by fetch(). Subclasses can
            override to add custom post-processing while calling super().
        """
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
        """
        Abstract method implementing the actual data retrieval logic.

        Subclasses must implement this method to define how data is retrieved
        for each flag. This is the core data access method that fetch() calls
        after configuration resolution and before post-processing.

        Args:
            flag: The validated flag to fetch data for
            effective_config: The fully resolved configuration for this operation
            **kwargs: Additional implementation-specific arguments

        Returns:
            DataFrame or Series containing the requested data. The returned
            data will be post-processed by _post_process_data() before being
            returned to the caller.

        Note:
            - This method should not perform flag validation (handled by fetch())
            - This method should not apply post-processing (handled separately)
            - This method should not handle caching (handled by fetch())

        Example:
            Implementing in a subclass::

                def _fetch(self, flag, effective_config, **kwargs):
                    if flag == 'generators':
                        return self.network.generators
                    elif flag == 'generators_t.p':
                        return self.network.generators_t.p
                    # ... handle other flags
        """
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
        """
        Fetch multiple flags and concatenate results into a single DataFrame.

        Convenience method for retrieving data from multiple flags and combining
        them into a single DataFrame with a MultiIndex. Useful for comparative
        analysis of multiple variables or creating wide-format data structures.

        Args:
            flags: Iterable of flags to fetch and concatenate
            concat_axis: Axis along which to concatenate (0=rows, 1=columns).
                Default is 1 (columns).
            concat_level_name: Name for the new MultiIndex level identifying
                the source flag. Default is 'variable'.
            concat_level_at_top: If True, the flag level is the outermost level
                in the MultiIndex. If False, it's moved to the innermost level.
                Default is True.
            config: Optional configuration override (see fetch() for details)
            **kwargs: Additional arguments passed to each fetch() call

        Returns:
            DataFrame with concatenated data and a MultiIndex identifying the
            source flag for each section

        Example:
            Fetching and comparing multiple variables::

                >>> # Fetch power output and efficiency for generators
                >>> combined = dataset.fetch_multiple_flags_and_concat(
                ...     flags=['generators_t.p', 'generators_t.efficiency'],
                ...     concat_level_name='metric'
                ... )
                >>> # Result has MultiIndex columns: (metric, generator_name)

                >>> # Row-wise concatenation
                >>> stacked = dataset.fetch_multiple_flags_and_concat(
                ...     flags=['bus_A_prices', 'bus_B_prices'],
                ...     concat_axis=0,
                ...     concat_level_name='bus'
                ... )

        See Also:
            - `fetch()` - Single flag data retrieval
            - `fetch_filter_groupby_agg()` - Fetch with filtering and aggregation
        """
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
        """
        Fetch data with model-based filtering, grouping, and aggregation.

        Provides a powerful one-line method for common data analysis patterns:
        filter time series by model properties, group by categories, and
        aggregate results. Requires a flag index with model mappings.

        Args:
            flag: Data flag to fetch (must have a linked model flag)
            model_filter_query: Pandas query string to filter based on model
                properties. Applied to the linked model DataFrame.
                Example: "carrier == 'solar'" or "p_nom > 100"
            prop_groupby: Model property or list of properties to group by.
                Adds these as MultiIndex levels and groups the data.
                Example: 'carrier' or ['carrier', 'bus']
            prop_groupby_agg: Aggregation function to apply after grouping.
                Standard pandas aggregation strings like 'sum', 'mean', 'max'.
                Only used if prop_groupby is specified.
            config: Optional configuration override (see fetch() for details)
            **kwargs: Additional arguments passed to fetch()

        Returns:
            Filtered and/or aggregated data. If prop_groupby is specified without
            prop_groupby_agg, returns a DataFrameGroupBy object.

        Raises:
            RuntimeError: If the flag has no linked model flag in the flag index

        Example:
            Common analysis patterns::

                >>> # Filter generators to only solar, sum by carrier
                >>> solar_gen = dataset.fetch_filter_groupby_agg(
                ...     'generators_t.p',
                ...     model_filter_query="carrier == 'solar'",
                ...     prop_groupby='carrier',
                ...     prop_groupby_agg='sum'
                ... )

                >>> # Group all generation by carrier and bus
                >>> by_carrier_bus = dataset.fetch_filter_groupby_agg(
                ...     'generators_t.p',
                ...     prop_groupby=['carrier', 'bus'],
                ...     prop_groupby_agg='sum'
                ... )

                >>> # Filter to large generators only
                >>> large_gens = dataset.fetch_filter_groupby_agg(
                ...     'generators_t.p',
                ...     model_filter_query="p_nom >= 500"
                ... )

        See Also:
            - `fetch()` - Basic data retrieval
            - [Pandas Utils](../utils/pandas_utils/index.md) - Underlying filter/group utilities
        """
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
        """
        Get the flag type class for this dataset type.

        Returns the type used for data flags in this dataset class. Subclasses
        can override to specify a custom flag type for type checking and
        validation.

        Returns:
            Type[FlagType]: The flag type class (default: FlagTypeProtocol)

        Note:
            Override in subclasses that use custom flag types.
        """
        from mesqual.flag.flag import FlagTypeProtocol
        return FlagTypeProtocol

    @classmethod
    def get_flag_index_type(cls) -> Type[FlagIndexType]:
        """
        Get the flag index type class for this dataset type.

        Returns the type used for the flag index in this dataset class.
        Subclasses can override to specify a custom flag index implementation.

        Returns:
            Type[FlagIndexType]: The flag index type class (default: FlagIndex)

        Note:
            Override in subclasses that use platform-specific flag indices.
        """
        from mesqual.flag.flag_index import FlagIndex
        return FlagIndex

    @classmethod
    def get_config_type(cls) -> Type[DatasetConfigType]:
        """
        Get the configuration type class for this dataset type.

        Returns the DatasetConfig subclass used by this dataset. Platform
        interfaces typically override this to return their extended config
        class with platform-specific options.

        Returns:
            Type[DatasetConfigType]: The config type class (default: DatasetConfig)

        Example:
            Creating a config instance for this dataset type::

                >>> ConfigClass = MyDataset.get_config_type()
                >>> config = ConfigClass(use_database=True)

        Note:
            Override in platform dataset subclasses to return platform-specific
            config types with additional options.
        """
        from mesqual.datasets.dataset_config import DatasetConfig
        return DatasetConfig

    @property
    def instance_config(self) -> DatasetConfigType:
        """
        Get the effective configuration for this dataset instance.

        Computes the merged configuration by combining:
        1. Base config defaults
        2. Class-level config (set via set_class_config)
        3. Instance-level config (passed to __init__ or set via set_instance_config)

        Later settings override earlier ones. This is the configuration used
        by fetch() unless overridden by a fetch-time config parameter.

        Returns:
            DatasetConfigType: The fully resolved configuration for this instance

        Example:
            Inspecting current configuration::

                >>> config = dataset.instance_config
                >>> print(config.use_database)
                True
                >>> print(config.auto_sort_datetime_index)
                True

        See Also:
            - `set_instance_config()` - Replace instance configuration
            - `set_class_config()` - Set class-level defaults
            - DatasetConfigManager - Configuration management system
        """
        from mesqual.datasets.dataset_config import DatasetConfigManager
        return DatasetConfigManager.get_effective_config(self.__class__, self._config)

    def set_instance_config(self, config: DatasetConfigType) -> None:
        """
        Replace the instance-level configuration for this dataset.

        Sets the configuration that will be merged with class-level defaults
        to produce the effective configuration used by fetch().

        Args:
            config: New configuration object to use for this instance

        Example:
            Setting a custom configuration::

                >>> from mesqual.datasets import DatasetConfig
                >>> config = DatasetConfig(use_database=False, auto_sort_datetime_index=False)
                >>> dataset.set_instance_config(config)

        See Also:
            - `instance_config` - Get the effective configuration
            - `set_instance_config_kwargs()` - Update individual settings
            - `set_class_config()` - Set class-level defaults
        """
        self._config = config

    def set_instance_config_kwargs(self, **kwargs) -> None:
        """
        Update individual configuration settings on this instance.

        Modifies specific attributes of the existing instance configuration
        without replacing the entire config object. Useful for tweaking
        individual settings.

        Args:
            **kwargs: Configuration attribute names and values to set

        Example:
            Adjusting specific settings::

                >>> dataset.set_instance_config_kwargs(
                ...     use_database=True,
                ...     auto_sort_datetime_index=False
                ... )

        Warning:
            Raises AttributeError if the config attribute doesn't exist.

        See Also:
            - `set_instance_config()` - Replace entire configuration
            - `instance_config` - Get the effective configuration
        """
        for key, value in kwargs.items():
            setattr(self._config, key, value)

    @classmethod
    def set_class_config(cls, config: DatasetConfigType) -> None:
        """
        Set the class-level configuration for all instances of this dataset type.

        Class-level configuration serves as the default for all instances of
        this class. Instance-level configuration (set via set_instance_config)
        can override these defaults.

        Args:
            config: Configuration object to use as class-level defaults

        Example:
            Setting defaults for all instances::

                >>> from mesqual.datasets import DatasetConfig
                >>> config = DatasetConfig(use_database=True)
                >>> MyDataset.set_class_config(config)
                >>>
                >>> # All new instances will use database by default
                >>> ds1 = MyDataset()  # uses database
                >>> ds2 = MyDataset()  # uses database

        Note:
            This affects all instances of the class, including existing ones
            that haven't overridden the setting at instance level.

        See Also:
            - `set_instance_config()` - Override for specific instances
            - DatasetConfigManager - Configuration management system
        """
        from mesqual.datasets.dataset_config import DatasetConfigManager
        DatasetConfigManager.set_class_config(cls, config)

    def __str__(self) -> str:
        return self.name

    def __hash__(self):
        return hash((self.name, self._config))
