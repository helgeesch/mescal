from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Type, TYPE_CHECKING, Union
import numpy as np

from mesqual.flag import FlagTypeProtocol
from mesqual.kpis.aggregations import Aggregation, ValueComparison, ArithmeticValueOperation
from mesqual.units import Units

if TYPE_CHECKING:
    from mesqual.datasets import Dataset


PRIMITIVE_VALUE_TYPES = Union[None, bool, int, float, str]


def _to_primitive(value: Any) -> PRIMITIVE_VALUE_TYPES:
    if value is None:
        return value
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)


@dataclass
class KPIAttributes:
    """
    Rich metadata container for KPI instances.

    Stores all context needed for filtering, grouping, visualization,
    and unit conversion of KPI values.

    Attributes:
        flag: Variable flag (e.g., 'BZ.Results.market_price')
        model_flag: Associated model flag (e.g., 'BZ.Model')
        object_name: Specific object identifier (e.g., 'DE-LU')
        aggregation: Aggregation applied (e.g., Aggregations.Mean)
        dataset_name: Name of the dataset
        dataset_type: Type of dataset ('scenario', 'comparison', etc.)
        value_comparison: Comparison operation for comparison KPIs
        arithmetic_operation: Arithmetic operation for derived KPIs
        reference_dataset_name: Reference dataset for comparisons
        variation_dataset_name: Variation dataset for comparisons
        name_prefix: Custom prefix for KPI name
        name_suffix: Custom suffix for KPI name
        custom_name: Complete custom name override
        unit: Physical unit of the KPI value
        target_unit: Target unit for conversion
        dataset_attributes: Additional attributes from dataset (e.g., scenario attributes)
        extra_attributes: Extra attributes set by user (e.g. for filtering / grouping purposes)
    """

    # Core identifiers
    flag: FlagTypeProtocol
    model_flag: FlagTypeProtocol | None = None
    object_name: Hashable | None = None
    aggregation: Aggregation | None = None

    # Dataset context
    dataset_name: str = ''
    dataset_type: Type[Dataset] = None

    # Comparison-specific
    value_comparison: ValueComparison | None = None
    arithmetic_operation: ArithmeticValueOperation | None = None
    reference_dataset_name: str | None = None
    variation_dataset_name: str | None = None

    # Naming
    name_prefix: str = ''
    name_suffix: str = ''
    custom_name: str | None = None

    # Unit handling
    unit: Units.Unit | None = None
    target_unit: Units.Unit | None = None

    # Additional attributes
    dataset_attributes: dict[str, Any] = field(default_factory=dict)
    extra_attributes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self, primitive_values: bool = True) -> dict[str, Any]:
        """
        Export attributes as dictionary for filtering.

        Args:
            primitive_values: If True, convert objects to strings for serialization

        Returns:
            Dictionary representation of attributes
        """
        d = {
            'flag': self.flag,
            'model_flag': self.model_flag,
            'object_name': self.object_name,
            'aggregation': self.aggregation,
            'dataset_name': self.dataset_name,
            'dataset_type': self.dataset_type,
            'value_comparison': self.value_comparison,
            'arithmetic_operation': self.arithmetic_operation,
            'reference_dataset_name': self.reference_dataset_name,
            'variation_dataset_name': self.variation_dataset_name,
            'name_prefix': self.name_prefix,
            'name_suffix': self.name_suffix,
            'custom_name': self.custom_name,
            'unit': self.unit,
            'target_unit': self.target_unit,
            **self.dataset_attributes,
            **self.extra_attributes
        }
        if primitive_values:
            d = {k: _to_primitive(v) for k, v in d.items()}
        return d

    def get(self, key: str, default: Any = None) -> Any:
        """
        Dict-like get interface for compatibility.

        Args:
            key: Attribute key to retrieve
            default: Default value if key not found

        Returns:
            Attribute value or default
        """
        return self.as_dict().get(key, default)
