from __future__ import annotations

from typing import TYPE_CHECKING, Union, Type, Iterable, Generic
from abc import ABC, abstractmethod

import pandas as pd

from mescal.typevars import DatasetConfigType, FlagType, FlagIndexType
from mescal.databases.database import Database
from mescal.utils.string_conventions import to_lower_snake
from mescal.flag.flag_index import EmptyFlagIndex
from mescal.utils.logging import get_logger

if TYPE_CHECKING:
    from mescal.datasets.dataset_collection import DatasetLinkCollection
    from mescal.kpis.kpi_collection import KPICollection
    from mescal.kpis.kpi_base import KPI, KPIFactory

logger = get_logger(__name__)


def flag_must_be_accepted(method):
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
        dataset.dotfetch.my.flag.as.string()
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
    def __init__(
            self,
            name: str = None,
            parent_dataset: Dataset = None,
            flag_index: FlagIndexType = None,
            attributes: dict = None,
            database: Database = None,
            config: DatasetConfigType = None
    ):
        self.name = name or f'{self.__class__.__name__}_{str(id(self))}'
        self._flag_index = flag_index or EmptyFlagIndex()
        self._parent_dataset = parent_dataset
        self._attributes: dict = attributes if attributes is not None else dict()
        self._database = database
        self._config = config
        self.dotfetch = _DotNotationFetcher(self)

        from mescal.kpis.kpi_collection import KPICollection
        self.kpi_collection: KPICollection = KPICollection()

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

    def add_kpis(self, kpis: Iterable[KPI | KPIFactory | Type[KPI]]):
        for kpi in kpis:
            self.add_kpi(kpi)

    def add_kpi(self, kpi: KPI | KPIFactory | Type[KPI]):
        from mescal.kpis.kpi_base import KPI
        from mescal.kpis.kpis_from_aggregations import KPIFactory
        if isinstance(kpi, KPIFactory):
            kpi = kpi.get_kpi(self)
        elif isinstance(kpi, type) and issubclass(kpi, KPI):
            kpi = kpi.from_factory(self)
        self.kpi_collection.add_kpi(kpi)

    def get_kpi_series(self, **kwargs) -> pd.Series:
        return self.kpi_collection.get_kpi_series(**kwargs)

    def clear_kpi_collection(self):
        from mescal.kpis import KPICollection
        self.kpi_collection = KPICollection()

    @property
    def attributes(self) -> pd.Series:
        att_series = pd.Series(self._attributes, name=self.name)
        return att_series

    def set_attributes(self, **kwargs):
        self._attributes.update(kwargs)

    @property
    def parent_dataset(self) -> 'DatasetLinkCollection':
        if self._parent_dataset is None:
            raise RuntimeError(f"Parent dataset called without / before assignment.")
        return self._parent_dataset

    @parent_dataset.setter
    def parent_dataset(self, parent_dataset: 'DatasetLinkCollection'):
        from mescal.datasets.dataset_collection import DatasetLinkCollection
        if not isinstance(parent_dataset, DatasetLinkCollection):
            raise TypeError(f"Parent parent_dataset must be of type {DatasetLinkCollection.__name__}")
        self._parent_dataset = parent_dataset

    @property
    @abstractmethod
    def accepted_flags(self) -> set[FlagType]:
        return set()

    def get_accepted_flags_containing_x(self, x: str, match_case: bool = False) -> set[FlagType]:
        """
        Returns a set with all flags that are accepted and whose string contain x.

        Example to retrieve all flags for generators:
        >>> ds = PyPSADataset()
        >>> ds.get_accepted_flags_containing_x('generators')
        {'generators', 'generators_t.p', 'generators_t.efficiency', ...}
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
        if config is None:
            return self.instance_config

        if isinstance(config, dict):
            temp_config = self.get_config_type()()
            temp_config.__dict__.update(config)
            return self.instance_config.merge(temp_config)

        from mescal.datasets.dataset_config import DatasetConfig
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

        from mescal.utils import pandas_utils

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
        from mescal.flag.flag import FlagTypeProtocol
        return FlagTypeProtocol

    @classmethod
    def get_flag_index_type(cls) -> Type[FlagIndexType]:
        from mescal.flag.flag_index import FlagIndex
        return FlagIndex

    @classmethod
    def get_config_type(cls) -> Type[DatasetConfigType]:
        from mescal.datasets.dataset_config import DatasetConfig
        return DatasetConfig

    @property
    def instance_config(self) -> DatasetConfigType:
        from mescal.datasets.dataset_config import DatasetConfigManager
        return DatasetConfigManager.get_effective_config(self.__class__, self._config)

    def set_instance_config(self, config: DatasetConfigType) -> None:
        self._config = config

    def set_instance_config_kwargs(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self._config, key, value)

    @classmethod
    def set_class_config(cls, config: DatasetConfigType) -> None:
        from mescal.datasets.dataset_config import DatasetConfigManager
        DatasetConfigManager.set_class_config(cls, config)

    @classmethod
    def _get_class_name_lower_snake(cls) -> str:
        return to_lower_snake(cls.__name__)

    def __str__(self) -> str:
        return self.name

    def __hash__(self):
        return hash((self.name, self._config))
