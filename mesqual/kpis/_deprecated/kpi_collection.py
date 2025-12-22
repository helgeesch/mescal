from __future__ import annotations

from typing import Iterable, Iterator
from collections import defaultdict

import pandas as pd

from mesqual._kpis_deprecated.kpi_base import KPI
from mesqual.utils.intersect_dicts import get_intersection_of_dicts
from mesqual.utils.logging import get_logger

logger = get_logger(__name__)


class KPICollection:
    def __init__(self, kpis: Iterable[KPI] = None, name: str = None):
        self._kpis: set[KPI] = set()
        if kpis:
            self.add_kpis(kpis)
        self._name = name

    @property
    def name(self) -> str:
        if self._name is not None:
            return self._name
        return f'{self.__class__.__name__} for ' + ' '.join(self.get_in_common_kpi_attributes(primitive_values=True))

    def add_kpis(self, kpis: Iterable[KPI]):
        for kpi in kpis:
            self.add_kpi(kpi)

    def add_kpi(self, kpi: KPI):
        if not isinstance(kpi, KPI):
            raise ValueError(f"Expected KPI instance, got {type(kpi)}")
        if kpi in self:
            logger.info(
                f"KPI with attributes \n"
                f"{kpi.attributes}\n "
                f"already exists in the collection and is not overwritten. \n"
                f"If you wish to overwrite, remove the kpi instance first or use a new instance (with a different id)."
            )
            return
        self._kpis.add(kpi)

    def compute_all(self, pbar: bool = False):
        if pbar:
            from tqdm import tqdm
            pbar = tqdm(self._kpis, total=self.size, desc=f'Computing KPIs {self.name}')
            for kpi in pbar:
                kpi.compute()
        else:
            for kpi in self._kpis:
                kpi.compute()

    def get_kpi_series(
            self,
            as_quantity: bool = False,
    ) -> pd.Series:
        if not as_quantity:
            return pd.Series(
                {kpi.name: kpi.value for kpi in self}
            )
        return pd.Series(
            {kpi.name: kpi.quantity for kpi in self}
        )

    def get_kpi_df_with_descriptive_attributes(
            self,
            unstack_column_levels: str | list[str] = None,
    ) -> pd.DataFrame:
        if self.empty:
            return pd.DataFrame()

        df = pd.concat(
            [
                kpi.get_kpi_as_series()
                for kpi in self._kpis
            ],
            axis=1
        ).T
        if unstack_column_levels:
            if isinstance(unstack_column_levels, str):
                unstack_column_levels = [unstack_column_levels]
            if not all(c in df.columns for c in unstack_column_levels):
                raise KeyError(f'Some of the unstack_column_levels where not found in the kpi')
            index_cols = [c for c in df.columns if c not in unstack_column_levels+['value', 'quantity']]
            df = df.pivot_table(
                index=index_cols,
                columns=unstack_column_levels,
                values=['value', 'quantity']
            )
        return df

    def get_kpi_by_attributes(self, attr_query: str = None, **kwargs) -> KPI:
        # TODO: write docstring
        subset = self.get_filtered_kpi_collection_by_attributes(attr_query=attr_query, **kwargs)
        num = len(subset._kpis)
        if num == 0:
            logger.warning(f'No KPI with matching attributes found.')
            return None
        elif num > 1:
            logger.warning(f'Found {num} KPIs for attributes {kwargs}. Returning the first one.')
        return next(iter(subset))

    def get_filtered_kpi_collection_by_attributes(self, attr_query: str = None, **kwargs) -> 'KPICollection':
        # TODO: write docstring
        filtered_kpi_collection = KPICollection()
        for kpi in self._kpis:
            if kpi.attributes.has_attr(attr_query=attr_query, **kwargs):
                filtered_kpi_collection.add_kpi(kpi)
        return filtered_kpi_collection

    def remove_kpi(self, kpi: KPI):
        self._kpis.remove(kpi)

    def remove_kpis_by_attributes(self, attr_query: str = None, **kwargs) -> 'KPICollection':
        subset = self.get_filtered_kpi_collection_by_attributes(attr_query=attr_query, **kwargs)
        num = len(subset._kpis)
        for kpi in subset:
            self.remove_kpi(kpi)
        logger.info(f'Removed {num} KPIs from Collection.')
        return self

    def __contains__(self, kpi: KPI) -> bool:
        return kpi in self._kpis

    def __iter__(self) -> Iterator[KPI]:
        for kpi in self._kpis:
            yield kpi

    def __len__(self) -> int:
        return len(self._kpis)

    @property
    def all_values(self) -> list[bool | int | float]:
        return [kpi.value for kpi in self._kpis]

    def get_in_common_kpi_attributes(self, primitive_values: bool = False) -> dict:
        dicts = [kpi.attributes.as_dict(primitive_values) for kpi in self]
        return get_intersection_of_dicts(dicts)

    def get_not_in_common_kpi_attributes_and_value_sets(self, primitive_values: bool = False) -> dict[str, list[bool | int | float | str]]:
        all_value_sets = self.get_all_kpi_attributes_and_value_sets(primitive_values)
        in_common_keys = self.get_in_common_kpi_attributes(primitive_values).keys()
        for k in in_common_keys:
            all_value_sets.pop(k, None)
        return all_value_sets

    def get_all_kpi_attributes_and_value_sets(self, primitive_values: bool = False) -> dict[str, list]:
        dicts = [kpi.attributes.as_dict(primitive_values) for kpi in self]
        all_keys = set([k for d in dicts for k in d.keys()])
        values = defaultdict(list)
        for d in dicts:
            for k in all_keys:
                v = d.get(k, None)
                if v not in values[k]:
                    values[k].append(v)
        return values

    def get_group_without(self, kpi: KPI) -> 'KPICollection':
        if kpi not in self._kpis:
            logger.warning(
                f'NotIntended: You are getting all kpis except a given one, '
                f'but the given one is not even in this {type(self)}. Sure this is intended?'
            )
        return KPICollection({k for k in self._kpis if not k == kpi})

    @property
    def empty(self) -> bool:
        return len(self._kpis) == 0

    @property
    def size(self) -> int:
        return len(self._kpis)
