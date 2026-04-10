"""
Dataset configuration system for controlling data processing behavior.

This module provides a hierarchical configuration system that allows fine-grained
control over how datasets fetch and process data. The system supports:

- **Default behavior** via the :class:`DatasetConfig` base class
- **Platform-specific extensions** by subclassing DatasetConfig (e.g., for PyPSA, PLEXOS)
- **Study-specific extensions** for custom processing logic
- **Class-level settings** via :class:`DatasetConfigManager`
- **Instance-level and fetch-time overrides** for maximum flexibility

Configuration Hierarchy:
    Configurations are resolved in a layered hierarchy, where later configs override earlier ones:

    1. **Base config**: Default values from the DatasetConfig class
    2. **Class config**: Set via ``DatasetConfigManager.set_class_config()``
    3. **Instance config**: Passed to ``Dataset.__init__(config=...)``
    4. **Fetch-time config**: Passed to ``Dataset.fetch(config=...)``

    Each layer only overrides non-None values, allowing partial overrides at any level.

Extending for Platform-Specific Behavior:
    To add platform-specific configuration options, subclass DatasetConfig:

        >>> from dataclasses import dataclass
        >>> from mesqual.datasets import DatasetConfig
        >>>
        >>> @dataclass
        >>> class MyPlatformConfig(DatasetConfig):
        >>>     # Platform-specific options
        >>>     convert_timestamps: bool = True
        >>>     reduce_resolution: bool = False
        >>>     custom_date_filter: list = None

    Then use it in your platform's Dataset implementation:

        >>> class MyPlatformDataset(Dataset):
        >>>     @classmethod
        >>>     def get_config_type(cls) -> Type[MyPlatformConfig]:
        >>>         return MyPlatformConfig

Study-Specific Extensions:
    For study-specific logic, you can further extend platform configs::

        >>> @dataclass
        >>> class MyStudyConfig(MyPlatformConfig):
        >>>     apply_custom_correction: bool = True

    And set it globally for the study::

        >>> from mesqual.datasets.dataset_config import DatasetConfigManager
        >>>
        >>> config = MyStudyConfig(apply_custom_correction=True)
        >>> DatasetConfigManager.set_class_config(MyStudyDataset, config)

Example Usage:
    Setting class-level config:

        >>> from mesqual.datasets.dataset_config import DatasetConfigManager
        >>> DatasetConfigManager.set_class_config(MyDataset, MyConfig(use_database=False))

    Override at fetch time::

        >>> # Using a dict for quick overrides
        >>> data = dataset.fetch('some_flag', config=dict(auto_sort_datetime_index=False))
        >>>
        >>> # Using a config object for type safety
        >>> custom_config = MyConfig(use_database=False, auto_sort_datetime_index=False)
        >>> data = dataset.fetch('some_flag', config=custom_config)
"""
from dataclasses import dataclass
from typing import Dict, Type, Optional, overload

from mesqual.typevars import DatasetConfigType
from mesqual.datasets.dataset import Dataset


class InvalidConfigSettingError(Exception):
    """Exception raised when an invalid configuration setting is provided.

    This exception is raised when a configuration value is incompatible
    with the expected type or constraints of a configuration option.
    """
    pass


@dataclass
class DatasetConfig:
    """
    Base configuration class for controlling Dataset behavior.

    DatasetConfig provides common configuration options that apply to all datasets
    in the MESQUAL framework. Platform-specific and study-specific configurations
    should extend this class to add additional options.

    The configuration system uses a merge-based hierarchy where each level can
    override settings from the previous level. The :meth:`merge` method combines
    configurations, with later values taking precedence over earlier ones.

    Attributes:
        use_database: If True, enables database caching for expensive fetch
            operations. When a database is configured on the dataset, fetched
            data will be cached and retrieved from cache on subsequent calls.
            Set to False to bypass caching. Default: True.

        auto_sort_datetime_index: If True, automatically sorts the returned
            DataFrame/Series by its DatetimeIndex after fetching. This ensures
            time-series data is always in chronological order regardless of
            the source data ordering. Default: True.

        remove_duplicate_indices: If True, automatically removes duplicate
            index entries from fetched data, keeping the first occurrence.
            A warning is logged when duplicates are found. This protects
            against data quality issues in source data. Default: True.

    Example:
        Creating a custom configuration::

            >>> config = DatasetConfig(use_database=False)
            >>> dataset = MyDataset(config=config)

        Extending for platform-specific options::

            >>> @dataclass
            ... class MyPlatformConfig(DatasetConfig):
            ...     custom_option: bool = True
            ...     date_filter: list = None
    """
    use_database: bool = True
    auto_sort_datetime_index: bool = True
    remove_duplicate_indices: bool = True

    def merge(self, other: Optional[DatasetConfigType | dict]) -> DatasetConfigType:
        """
        Merge this configuration with another, returning a new combined config.

        Creates a new configuration instance that combines settings from both
        configurations. Values from ``other`` override values from ``self``,
        but only for non-None values. This allows partial overrides where you
        only specify the settings you want to change.

        The merge creates a new instance of the same type as ``self``, ensuring
        that subclass-specific attributes are preserved.

        Args:
            other: Configuration to merge with. Can be:
                - None: Returns self unchanged
                - dict: Keys map to config attribute names
                - DatasetConfig: Another config instance (same or subclass)

        Returns:
            A new configuration instance combining both configs. The return
            type matches the type of ``self``.

        Example:

            >>> base = DatasetConfig(use_database=True, auto_sort_datetime_index=True)
            >>> override = DatasetConfig(use_database=False)
            >>> merged = base.merge(override)
            >>> merged.use_database
                False
            >>> merged.auto_sort_datetime_index  # Preserved from base
                True

            Using a dict for quick overrides:

                >>> merged = base.merge({'use_database': False})
        """
        if other is None:
            return self

        merged_config = self.__class__()

        for attr_name in dir(self):
            if not attr_name.startswith('_'):  # Skip private attributes
                setattr(merged_config, attr_name, getattr(self, attr_name))

        if isinstance(other, dict):
            for key, value in other.items():
                if value is not None:
                    setattr(merged_config, key, value)
            return merged_config

        for attr_name in dir(other):
            if not attr_name.startswith('_'):
                other_value = getattr(other, attr_name)
                if other_value is not None:
                    setattr(merged_config, attr_name, other_value)

        return merged_config

    def __repr__(self) -> str:
        """Return a string representation showing all config attributes."""
        attrs = {
            name: getattr(self, name)
            for name in dir(self)
            if not name.startswith('_') and not callable(getattr(self, name))
        }
        return f"{self.__class__.__name__}({attrs})"


class DatasetConfigManager:
    """
    Registry for managing class-level dataset configurations.

    DatasetConfigManager provides a centralized way to set default configurations
    for entire dataset classes. This is useful for establishing project-wide or
    study-wide defaults that apply to all instances of a particular dataset type.

    The manager maintains an internal registry mapping dataset classes to their
    configurations. When a dataset resolves its effective configuration, it
    queries this registry to find any class-level settings.

    Configuration Resolution Order:
        1. Base defaults from ``dataset_class.get_config_type()()``
        2. Class config from ``DatasetConfigManager._class_configs``
        3. Instance config passed to dataset constructor
        4. Fetch-time config passed to ``.fetch()``

    Class Attributes:
        _class_configs: Internal registry mapping Dataset classes to their
            configured DatasetConfig instances.

    Example:
        Setting a global config for all instances of a dataset class:

            >>> from mesqual.datasets.dataset_config import DatasetConfigManager
            >>>
            >>> # Disable caching for all MyDataset instances
            >>> DatasetConfigManager.set_class_config(
            ...     MyDataset,
            ...     MyConfig(use_database=False)
            ... )

        Updating specific settings on an existing class config:

            >>> # First set a base config
            >>> DatasetConfigManager.set_class_config(MyDataset, MyConfig())
            >>>
            >>> # Later update just one setting
            >>> DatasetConfigManager.update_class_config_kwargs(
            ...     MyDataset,
            ...     auto_sort_datetime_index=False
            ... )

    Note:
        Class configs are shared across all instances. Changes made via
        :meth:`update_class_config_kwargs` will affect all existing and
        future instances of that dataset class.
    """
    _class_configs: Dict[Type[Dataset], DatasetConfig] = {}

    @classmethod
    @overload
    def set_class_config(cls, dataset_class: Type[Dataset], config: DatasetConfigType) -> None:
        ...

    @classmethod
    def set_class_config(cls, dataset_class: Type[Dataset], config: DatasetConfig) -> None:
        """
        Set the class-level configuration for a dataset type.

        This configuration will be applied to all instances of the specified
        dataset class as part of the configuration resolution hierarchy.

        Args:
            dataset_class: The Dataset class to configure.
            config: Configuration instance to use as the class default.

        Example:

            >>> from mesqual.datasets.dataset_config import DatasetConfigManager
            >>>
            >>> config = MyPlatformConfig(
            ...     use_database=False,
            ...     convert_timestamps=True
            ... )
            >>> DatasetConfigManager.set_class_config(MyPlatformDataset, config)
        """
        cls._class_configs[dataset_class] = config

    @classmethod
    @overload
    def update_class_config_kwargs(cls, dataset_class: Type[Dataset], **config_kwargs) -> None:
        ...

    @classmethod
    def update_class_config_kwargs(cls, dataset_class: Type[Dataset], **config_kwargs) -> None:
        """
        Update specific settings on an existing class configuration.

        Modifies the class-level config in place. A class config must already
        exist for this dataset class (set via :meth:`set_class_config`).

        This is useful for making targeted adjustments without replacing the
        entire configuration.

        Args:
            dataset_class: The Dataset class whose config to update.
            **config_kwargs: Config attribute names and their new values.

        Raises:
            KeyError: If no class config exists for the dataset class.

        Example:

            >>> # Assuming a class config already exists
            >>> DatasetConfigManager.update_class_config_kwargs(
            ...     MyDataset,
            ...     use_database=False,
            ...     auto_sort_datetime_index=False
            ... )

        Warning:
            This modifies the config in place, affecting all instances
            immediately. Use with caution in multi-instance scenarios.
        """
        for k, v in config_kwargs.items():
            setattr(cls._class_configs[dataset_class], k, v)

    @classmethod
    @overload
    def get_effective_config(
            cls,
            dataset_class: Type[Dataset],
            instance_config: Optional[DatasetConfigType] = None
    ) -> DatasetConfigType:
        ...

    @classmethod
    def get_effective_config(
            cls,
            dataset_class: Type[Dataset],
            instance_config: Optional[DatasetConfig] = None
    ) -> DatasetConfig:
        """
        Resolve the effective configuration for a dataset class.

        Combines the base config, class config, and optional instance config
        using the merge hierarchy. Each level overrides non-None values from
        the previous level.

        Resolution order:
            1. Base config from ``dataset_class.get_config_type()()``
            2. Class config from internal registry (if set)
            3. Instance config parameter (if provided)

        Args:
            dataset_class: The Dataset class to get config for.
            instance_config: Optional instance-level config to merge.

        Returns:
            The fully resolved configuration combining all levels.
            The return type matches the dataset's config type.

        Example:

            >>> # Get effective config without instance override
            >>> config = DatasetConfigManager.get_effective_config(MyDataset)
            >>>
            >>> # Get effective config with instance override
            >>> instance_cfg = MyConfig(use_database=False)
            >>> config = DatasetConfigManager.get_effective_config(
            ...     MyDataset,
            ...     instance_config=instance_cfg
            ... )
        """
        config_type = dataset_class.get_config_type()
        base_config = config_type()

        # Walk MRO from most generic to most specific, merging class configs
        for klass in reversed(dataset_class.__mro__):
            class_config = cls._class_configs.get(klass)
            if class_config:
                base_config = base_config.merge(class_config)

        if instance_config:
            base_config = base_config.merge(instance_config)

        return base_config
