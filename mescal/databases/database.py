from abc import ABC, abstractmethod
from typing import Generic

import pandas as pd

from mescal.typevars import DatasetType, FlagType, DatasetConfigType


class Database(Generic[DatasetType, DatasetConfigType], ABC):
    @abstractmethod
    def get(
            self,
            dataset: DatasetType,
            flag: FlagType,
            config: DatasetConfigType,
            **kwargs
    ) -> pd.Series | pd.DataFrame:
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
        pass
    
    @abstractmethod
    def key_is_up_to_date(
            self,
            dataset: DatasetType,
            flag: FlagType,
            config: DatasetConfigType,
            **kwargs
    ):
        pass
