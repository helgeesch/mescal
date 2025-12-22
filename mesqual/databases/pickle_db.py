from typing import Optional, List
import os
import glob
from pathlib import Path

import pandas as pd

from mesqual.typevars import DatasetType, FlagType, DatasetConfigType
from mesqual.databases.database import Database


class PickleDatabase(Database):
    """File-based database implementation using pickle format.

    This database stores each cached dataset as a separate pickle file with
    a filename generated from hashed dataset, flag, config, and kwargs parameters.
    Files are organized in a single directory structure.

    Attributes:
        folder_path (Path): Path to the directory containing pickle files

    Example:
        Basic usage:

        >>> db = PickleDatabase("/path/to/cache")
        >>> # Files will be stored as: dataset_flag_config_hash_kwargs_hash.pickle
        >>> db.set(dataset, flag, config, my_dataframe)
        >>> data = db.get(dataset, flag, config)

        Cleanup operations:

        >>> # Delete all data for a specific dataset
        >>> db.delete(dataset=my_dataset)
        >>> # List available keys
        >>> keys = db.list_keys()
    """
    def __init__(self, folder_path: str):
        """Initialize the pickle database.

        Args:
            folder_path: Path to directory where pickle files will be stored.
                        Directory will be created if it doesn't exist.
        """
        self._folder_path = folder_path
        self._ensure_folder_exists(folder_path)

    @property
    def folder_path(self) -> Path:
        return Path(self._folder_path)
    
    def get(
            self,
            dataset: DatasetType,
            flag: FlagType,
            config: DatasetConfigType,
            **kwargs
    ) -> pd.Series | pd.DataFrame:
        """Retrieve data from pickle file.

        Args:
            dataset: Dataset type identifier
            flag: Processing flag or stage identifier
            config: Configuration object containing processing parameters
            **kwargs: Additional keyword arguments for cache key generation

        Returns:
            pd.Series | pd.DataFrame: The cached data loaded from pickle file

        Raises:
            FileNotFoundError: If no pickle file exists for the given key combination
        """
        return pd.read_pickle(self._get_file_path(dataset, flag, config, **kwargs))

    def set(
            self,
            dataset: DatasetType,
            flag: FlagType,
            config: DatasetConfigType,
            value,
            **kwargs
    ):
        """Store data as pickle file.

        Args:
            dataset: Dataset type identifier
            flag: Processing flag or stage identifier
            config: Configuration object containing processing parameters
            value: Data to store (pandas Series or DataFrame)
            **kwargs: Additional keyword arguments for cache key generation
        """
        file_path = self._get_file_path(dataset, flag, config, **kwargs)
        value.to_pickle(file_path)

    def key_is_up_to_date(
            self,
            dataset: DatasetType,
            flag: FlagType,
            config: DatasetConfigType,
            **kwargs
    ):
        """Check if pickle file exists for the given key.

        Args:
            dataset: Dataset type identifier
            flag: Processing flag or stage identifier
            config: Configuration object containing processing parameters
            **kwargs: Additional keyword arguments for cache key generation

        Returns:
            bool: True if pickle file exists for the given key combination
        """
        file_path = self._get_file_path(dataset, flag, config, **kwargs)
        return os.path.exists(file_path)

    def _get_config_hash(self, config: DatasetConfigType = None) -> str:
        """Generate hash for configuration object.

        Args:
            config: Configuration object to hash (None returns empty string)

        Returns:
            str: Hash string representing the configuration, or empty string if None
        """
        if config is None:
            return ""

        attrs = {
            name: getattr(config, name)
            for name in dir(config)
            if not name.startswith('_') and not callable(getattr(config, name))
        }

        sorted_items = sorted(attrs.items())

        # Convert to string representation for hashing
        config_str = str(sorted_items)
        return str(hash(config_str))

    def _get_kwargs_hash(self, kwargs: dict) -> str:
        """Generate hash for keyword arguments dictionary.

        Args:
            kwargs: Dictionary of keyword arguments to hash

        Returns:
            str: Hash string representing the kwargs, or empty string if empty
        """
        if not kwargs:
            return ""

        str_dict = {
            str(k): str(v)
            for k, v in kwargs.items()
        }

        sorted_items = sorted(str_dict.items())
        return str(hash(str(sorted_items)))

    def _get_file_path(self, dataset: DatasetType, flag: FlagType, config: DatasetConfigType = None, **kwargs) -> str:
        """Generate file path for the given parameters.

        Args:
            dataset: Dataset type identifier
            flag: Processing flag identifier
            config: Configuration object (optional)
            **kwargs: Additional keyword arguments

        Returns:
            str: Full file path for the pickle file
        """
        components = [dataset.name, str(flag)]

        config_hash = self._get_config_hash(config)
        if config_hash:
            components.append(f"config_{config_hash}")

        kwargs_hash = self._get_kwargs_hash(kwargs)
        if kwargs_hash:
            components.append(f"kwargs_{kwargs_hash}")

        filename = "_".join(components) + ".pickle"
        return os.path.join(self._folder_path, filename)

    def delete(
            self,
            dataset: Optional[DatasetType] = None,
            flag: Optional[FlagType] = None,
            config: Optional[DatasetConfigType] = None,
            **kwargs
    ):
        """Delete pickle files matching the given criteria.

        Uses glob patterns to match and delete files. If no parameters are provided,
        this will delete all pickle files in the database directory.

        Args:
            dataset: Dataset type to filter by (None matches all datasets)
            flag: Flag to filter by (None matches all flags)
            config: Configuration to filter by (None matches all configs)
            **kwargs: Additional keyword arguments to filter by

        Note:
            Be careful when calling without parameters as it will delete all data.
        """
        pattern_parts = []

        if dataset is not None:
            pattern_parts.append(dataset.name)
        else:
            pattern_parts.append("*")

        if flag is not None:
            pattern_parts.append(str(flag))
        else:
            pattern_parts.append("*")

        # Handle config and kwargs patterns
        if config is not None or kwargs:
            config_hash = self._get_config_hash(config) if config is not None else "*"
            kwargs_hash = self._get_kwargs_hash(kwargs) if kwargs else "*"

            if config is not None and config_hash:
                pattern_parts.append(f"config_{config_hash}")
            elif config is None:
                pattern_parts.append("*")

            if kwargs and kwargs_hash:
                pattern_parts.append(f"kwargs_{kwargs_hash}")
            elif not kwargs:
                pattern_parts.append("*")

        pattern = "_".join(pattern_parts) + ".pickle"
        file_pattern = os.path.join(self._folder_path, pattern)

        for file_path in glob.glob(file_pattern):
            os.remove(file_path)

    def list_keys(
            self,
            dataset: Optional[DatasetType] = None,
            flag: Optional[FlagType] = None
    ) -> List[str]:
        """List all cache keys matching the given criteria.

        Args:
            dataset: Dataset type to filter by (None matches all datasets)
            flag: Flag to filter by (None matches all flags)

        Returns:
            List[str]: Sorted list of cache keys (filenames without .pickle extension)
                      matching the specified criteria
        """
        pattern_parts = []

        if dataset is not None:
            pattern_parts.append(dataset.name)
        else:
            pattern_parts.append("*")

        if flag is not None:
            pattern_parts.append(str(flag))
        else:
            pattern_parts.append("*")

        pattern_parts.append("*")  # For remaining components
        pattern = "_".join(pattern_parts) + ".pickle"
        file_pattern = os.path.join(self._folder_path, pattern)

        keys = []
        for file_path in glob.glob(file_pattern):
            filename = os.path.basename(file_path)
            key = filename.replace(".pickle", "")
            keys.append(key)

        return sorted(keys)

    @staticmethod
    def _ensure_folder_exists(folder_path: str):
        """Create directory if it doesn't exist.

        Args:
            folder_path: Path to directory to create
        """
        os.makedirs(folder_path, exist_ok=True)
