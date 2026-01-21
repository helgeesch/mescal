from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
import pandas as pd

from mesqual.datasets import DatasetComparison
from mesqual.flag import FlagTypeProtocol
from mesqual.units import Units
from mesqual.kpis.attributes import KPIAttributes

if TYPE_CHECKING:
    from mesqual.datasets.dataset import Dataset


@dataclass
class KPI:
    """
    Single computed KPI with value and rich metadata.

    Represents a scalar metric (e.g., "Mean market price for BZ DE-LU in scenario base")
    with all context needed for filtering, visualization, and unit conversion.

    Attributes:
        value: The computed scalar value
        attributes: Rich metadata container
        dataset: Reference to source dataset for lazy model access
    """

    value: float | int
    attributes: KPIAttributes
    dataset: Dataset

    _object_info: pd.Series | None = None
    _quantity: Units.Quantity | None = None

    @property
    def quantity(self) -> Units.Quantity:
        """
        Return value as pint Quantity with unit.

        Returns:
            Quantity with value and unit
        """
        if self._quantity is None:
            unit = self.attributes.unit or self.dataset.flag_index.get_unit(self.attributes.flag)
            self._quantity = self.value * unit
        return self._quantity

    @property
    def name(self) -> str:
        """
        Generate human-readable unique name.

        Returns:
            KPI name (custom or auto-generated)
        """
        if self.attributes.custom_name:
            return self.attributes.custom_name

        return self._generate_automatic_name()

    def _generate_automatic_name(self) -> str:
        """
        Auto-generate name from attributes.

        Format: [prefix] flag aggregation object [comparison/operation] [suffix]

        Returns:
            Generated name string
        """
        parts = []

        parts.append(self.attributes.dataset_name)

        if self.attributes.name_prefix:
            parts.append(self.attributes.name_prefix)

        # Core components
        flag_short = str(self.attributes.flag)
        parts.append(flag_short)

        if self.attributes.aggregation:
            parts.append(str(self.attributes.aggregation))

        if self.attributes.object_name:
            parts.append(str(self.attributes.object_name))

        if self.attributes.value_comparison:
            parts.append(str(self.attributes.value_comparison))

        if self.attributes.arithmetic_operation:
            parts.append(str(self.attributes.arithmetic_operation))

        if self.attributes.name_suffix:
            parts.append(self.attributes.name_suffix)

        return ' '.join(parts)

    def get_object_info_from_model(self) -> pd.Series:
        """
        Lazy fetch of object metadata from Model flag.

        Returns:
            Series with object properties from model DataFrame
        """
        if self._object_info is None:
            if self.attributes.model_flag and self.attributes.object_name:
                if isinstance(self.dataset, DatasetComparison):
                    sets = [self.dataset.variation_dataset, self.dataset.reference_dataset]
                else:
                    sets = [self.dataset]
                for ds in sets:
                    if ds.flag_is_accepted(self.attributes.model_flag):
                        model_df = ds.fetch(self.attributes.model_flag)
                        if self.attributes.object_name in model_df.index:
                            self._object_info = model_df.loc[self.attributes.object_name]
                            break
                if self._object_info is None:
                    self._object_info = pd.Series()
        return self._object_info

    def get_kpi_name_with_dataset_name(self) -> str:
        """
        Get KPI name with dataset name for visualization tooltips.

        Returns:
            Name including dataset (e.g., "market_price Mean [base_scenario]")
        """
        base_name = self.name
        if self.attributes.dataset_name:
            return f"{base_name} [{self.attributes.dataset_name}]"
        return base_name

    def to_dict(self, primitive_values: bool = True) -> dict:
        """
        Export KPI as flat dictionary for tables.

        Args:
            primitive_values: If True, convert objects to strings for serialization

        Returns:
            Dictionary with all KPI data
        """
        return {
            'name': self.name,
            'value': self.value,
            'quantity': str(self.quantity) if primitive_values else self.quantity,
            **self.attributes.as_dict(primitive_values=primitive_values),
        }

    def __repr__(self) -> str:
        """String representation of KPI."""
        return f"KPI(name='{self.name}', value={self.value}, unit={self.quantity.units})"
