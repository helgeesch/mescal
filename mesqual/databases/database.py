"""Abstract database interface for MESQUAL framework.

This module defines the abstract Database class that provides a unified interface
for different storage backends in the MESQUAL energy modeling framework.
"""

from abc import ABC, abstractmethod
from typing import Generic, Optional, List

import pandas as pd

from mesqual.typevars import DatasetType, FlagType, DatasetConfigType


class Database(Generic[DatasetType, DatasetConfigType], ABC):
    """Abstract base class for database implementations.

    This class defines the interface that all database implementations must follow.
    It provides methods for storing, retrieving, and managing cached dataset results
    in the MESQUAL framework.

    The database uses a key-based system where each entry is identified by:
    - dataset: The type of dataset (e.g., network topology, time series)
    - flag: Processing stage or variant identifier
    - config: Configuration object with processing parameters
    - kwargs: Additional keyword arguments for cache differentiation

    Generic Types:
        DatasetType: Type representing different dataset categories
        DatasetConfigType: Type for configuration objects
    """
    @abstractmethod
    def get(
            self,
            dataset: DatasetType,
            flag: FlagType,
            config: DatasetConfigType,
            **kwargs
    ) -> pd.Series | pd.DataFrame:
        """Retrieve data from the database.

        Args:
            dataset: Dataset type identifier
            flag: Processing flag or stage identifier
            config: Configuration object containing processing parameters
            **kwargs: Additional keyword arguments for cache key generation

        Returns:
            pd.Series | pd.DataFrame: The cached data

        Raises:
            KeyError: If no data is found for the given key combination
        """
        pass
    
    @abstractmethod
    def set(
            self,
            dataset: DatasetType,
            flag: FlagType,
            config: DatasetConfigType,
            value,
            **kwargs
    ):
        """Store data in the database.

        Args:
            dataset: Dataset type identifier
            flag: Processing flag or stage identifier
            config: Configuration object containing processing parameters
            value: Data to store (pandas Series or DataFrame)
            **kwargs: Additional keyword arguments for cache key generation
        """
        pass
    
    @abstractmethod
    def key_is_up_to_date(
            self,
            dataset: DatasetType,
            flag: FlagType,
            config: DatasetConfigType,
            **kwargs
    ):
        """Check if cached data exists and is up to date.

        Args:
            dataset: Dataset type identifier
            flag: Processing flag or stage identifier
            config: Configuration object containing processing parameters
            **kwargs: Additional keyword arguments for cache key generation

        Returns:
            bool: True if data exists for the given key combination
        """
        pass

    @abstractmethod
    def delete(
            self,
            dataset: Optional[DatasetType] = None,
            flag: Optional[FlagType] = None,
            config: Optional[DatasetConfigType] = None,
            **kwargs
    ):
        """Delete data matching the given criteria.

        Args:
            dataset: Dataset type to filter by (None for all datasets)
            flag: Flag to filter by (None for all flags)
            config: Configuration to filter by (None for all configs)
            **kwargs: Additional keyword arguments to filter by

        Note:
            If no parameters are provided, this could delete all data.
            Use with caution.
        """
        pass

    @abstractmethod
    def list_keys(
            self,
            dataset: Optional[DatasetType] = None,
            flag: Optional[FlagType] = None
    ) -> List[str]:
        """List all cache keys matching the given criteria.

        Args:
            dataset: Dataset type to filter by (None for all datasets)
            flag: Flag to filter by (None for all flags)

        Returns:
            List[str]: Sorted list of cache keys matching the criteria
        """
        pass
