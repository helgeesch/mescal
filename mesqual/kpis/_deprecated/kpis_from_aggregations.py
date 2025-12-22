from __future__ import annotations

from typing import Hashable, Generic, TYPE_CHECKING

import pandas as pd

from mesqual.units import Units
from mesqual.typevars import DatasetType, FlagType
from mesqual.utils.pandas_utils.filter import filter_by_model_query
from mesqual._kpis_deprecated.kpi_base import KPI, KPIFactory, KPIAttributes
from mesqual.utils.logging import get_logger

if TYPE_CHECKING:
    from mesqual._kpis_deprecated.aggs import Aggregation

logger = get_logger(__name__)

SPACE = ' '


class MultipleColumnsInSubsetException(Exception):
    pass


class NoColumnDefinedException(Exception):
    pass


class MissingColumnsFromSubsetException(Exception):
    pass


class FlagAggKPI(Generic[DatasetType], KPI):

    def __init__(
            self,
            dataset: DatasetType,
            flag: FlagType,
            aggregation: Aggregation,
            column_subset: Hashable | list[Hashable] = None,
            model_query: str = None,
            kpi_name_prefix: str = None,
            kpi_name_suffix: str = None,
            kpi_name: str = None,
    ):
        self._flag = flag
        self._aggregation = aggregation
        self._column_subset = column_subset
        self._model_query = model_query
        self._kpi_name_prefix = kpi_name_prefix
        self._kpi_name_suffix = kpi_name_suffix
        self._kpi_name = kpi_name

        super().__init__(dataset=dataset)

    @property
    def flag(self) -> FlagType:
        return self._flag

    @property
    def aggregation(self) -> Aggregation:
        return self._aggregation

    @property
    def column_subset(self) -> Hashable | list[Hashable]:
        return self._column_subset

    @property
    def model_query(self) -> str:
        return self._model_query

    @property
    def kpi_name_prefix(self) -> str:
        return self._kpi_name_prefix

    @property
    def kpi_name_suffix(self) -> str:
        return self._kpi_name_suffix

    @property
    def kpi_name(self) -> str:
        return self._kpi_name

    def _get_kpi_attributes(self) -> KPIAttributes:
        atts = super()._get_kpi_attributes()
        atts.flag = self._flag
        atts.aggregation = self._aggregation
        atts.column_subset = self._column_subset
        atts.model_query = self._model_query
        atts.name_prefix = self._kpi_name_prefix
        atts.name_suffix = self._kpi_name_suffix

        try:
            atts.object_name = self.get_attributed_object_name()
        except (NotImplementedError, NoColumnDefinedException, MultipleColumnsInSubsetException):
            pass
        try:
            atts.model_flag = self.get_attributed_model_flag()
        except (NotImplementedError,):
            pass

        return atts

    @property
    def name(self) -> str:
        if self._kpi_name:
            return self._kpi_name
        return self._generic_kpi_name_generator(
            aggregation=self._aggregation,
            flag=self._flag,
            column_subset=self._column_subset,
            model_query=self._model_query,
            kpi_name_prefix=self._kpi_name_prefix,
            kpi_name_suffix=self._kpi_name_suffix,
        )

    @staticmethod
    def _generic_kpi_name_generator(
            flag: FlagType,
            aggregation: Aggregation,
            column_subset: Hashable | list[Hashable] = None,
            model_query: str = None,
            kpi_name_prefix: str = None,
            kpi_name_suffix: str = None,
    ) -> str:
        name = ''

        if kpi_name_prefix is not None:
            name += kpi_name_prefix + SPACE

        name += f'{aggregation} {flag}'

        if model_query is not None:
            name += f' ("{model_query}")'
        if column_subset is not None:
            name += f' ({column_subset})'
        if kpi_name_suffix is not None:
            name += SPACE + kpi_name_suffix

        return name

    @property
    def unit(self) -> Units.Unit:
        if self._aggregation.unit is not None:
            return self._aggregation.unit
        return self._dataset.flag_index.get_unit(self._flag)

    def _fetch_filtered_data(self, dataset: DatasetType) -> pd.DataFrame:
        from mesqual.datasets.dataset_comparison import DatasetComparison
        flag = self._flag
        if isinstance(dataset, DatasetComparison):
            data = dataset.fetch(flag, fill_value=0)
        else:
            data = dataset.fetch(flag)

        if self._model_query:
            model_flag = dataset.flag_index.get_linked_model_flag(flag)
            model_df = dataset.fetch(model_flag)
            data = filter_by_model_query(data, model_df, self._model_query)

        if self._column_subset:
            subset = self._get_column_subset_as_list()
            if not set(subset).issubset(data.columns):
                missing = set(subset).difference(data.columns)
                raise MissingColumnsFromSubsetException(
                    f'Trying to fetch the data for {self.get_kpi_name_with_dataset_name()}, '
                    f'the following columns were not found in the result_df: {missing}.'
                )
            data = data[subset]

        return data

    def _get_column_subset_as_list(self) -> list[str | int]:
        return self._column_subset if isinstance(self._column_subset, list) else [self._column_subset]

    def get_attributed_object_name(self) -> str | int:
        """
        Only necessary in case one wants to be able to retrieve the attributed object_name for a KPI.
        For example you'd want this in case you need this info for plotting.
        """
        if self._column_subset is None:
            raise NoColumnDefinedException(
                f"You are trying to get the attributed_object_name for KPI {self.get_kpi_name_with_dataset_name()}. "
                f"However, this method is only valid if you define exactly 1 column in the column_subset. "
                f"Currently, the column_subset is not set at all (None)."
            )

        subset = self._get_column_subset_as_list()
        if len(subset) > 1:
            raise MultipleColumnsInSubsetException(
                f"You are trying to get the attributed_object_name for KPI {self.get_kpi_name_with_dataset_name()}. "
                f"However, This method is only valid if you define exactly 1 column in the column_subset. "
                f"Currently, the column_subset contains multiple columns: {self._column_subset}."
            )

        column = subset[0]
        if isinstance(column, tuple):
            model_flag = self.get_attributed_model_flag()
            model_df = self._dataset.fetch(model_flag)
            options = set(column).intersection(model_df.index)
            if len(options) == 1:
                return list(options)[0]
            elif len(options) > 1:
                selected_object = [i for i in column if i in options][0]
                logger.warning(
                    f"Column Subset appears to be a tuple ({column}) with multiple possible primary objects: {options}."
                    f"We are proceeding with the first one ({selected_object})"
                )
                return selected_object
        else:
            return column

    def get_attributed_model_flag(self) -> FlagType:
        """
        Only necessary in case one wants to be able to retrieve the attributed model_flag for a KPI.
        For example you'd want this for applying a model query, or in case you need this info for plotting.
        """
        return self._dataset.flag_index.get_linked_model_flag(self._flag)

    def compute(self):
        data = self._fetch_filtered_data(self._dataset)
        self._value = self._aggregation(data)
        self._has_been_computed = True

    def required_flags(self) -> set[FlagType]:
        flags = {self._flag}
        if self._model_query:
            flags.add(self.get_attributed_model_flag())
        return flags


class FlagAggKPIFactory(Generic[DatasetType], KPIFactory[DatasetType, FlagAggKPI]):
    def __init__(
            self,
            flag: FlagType,
            aggregation: Aggregation,
            column_subset: Hashable | list[Hashable] = None,
            model_query: str = None,
            kpi_name_prefix: str = None,
            kpi_name_suffix: str = None,
            kpi_name: str = None,
    ):
        self._flag = flag
        self._aggregation = aggregation
        self._column_subset = column_subset
        self._model_query = model_query
        self._kpi_name_prefix = kpi_name_prefix
        self._kpi_name_suffix = kpi_name_suffix
        self._kpi_name = kpi_name

    def get_kpi_class(self) -> type[FlagAggKPI]:
        return FlagAggKPI

    def get_kpi(self, dataset: DatasetType) -> FlagAggKPI:
        kpi = self.get_kpi_class()(
            dataset=dataset,
            flag=self._flag,
            aggregation=self._aggregation,
            column_subset=self._column_subset,
            model_query=self._model_query,
            kpi_name_prefix=self._kpi_name_prefix,
            kpi_name_suffix=self._kpi_name_suffix,
            kpi_name=self._kpi_name,
        )
        return kpi

    @property
    def flag(self):
        return self._flag

    @property
    def aggregation(self):
        return self._aggregation

    @property
    def column_subset(self):
        return self._column_subset

    @property
    def model_query(self):
        return self._model_query

    @property
    def kpi_name_prefix(self):
        return self._kpi_name_prefix

    @property
    def kpi_name_suffix(self):
        return self._kpi_name_suffix

    @property
    def kpi_name(self):
        return self._kpi_name
