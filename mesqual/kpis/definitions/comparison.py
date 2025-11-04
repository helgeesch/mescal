from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
import pandas as pd

from mesqual.datasets.dataset_comparison import DatasetComparison
from mesqual.kpis.definitions.base import KPIDefinition
from mesqual.kpis.kpi import KPI
from mesqual.kpis.attributes import KPIAttributes
from mesqual.kpis.aggregations import ValueComparison

if TYPE_CHECKING:
    from mesqual.typevars import FlagTypeProtocol


@dataclass
class ComparisonKPIDefinition(KPIDefinition):
    """
    KPI definition for comparing values between two datasets.

    Creates KPIs by:
    1. Generating base KPIs from base_definition for reference and variation datasets
    2. Applying comparison operation to corresponding KPI values
    3. Creating comparison KPI instances with proper metadata

    This is typically used to compare scenarios (e.g., "Price increase in
    high_res vs base scenario").

    Attributes:
        base_definition: Definition to generate base KPIs from
        comparison: Comparison operation to apply (e.g., ValueComparisons.Increase)
        name_prefix: Optional prefix for comparison KPI names
        name_suffix: Optional suffix for comparison KPI names
    """

    base_definition: KPIDefinition
    comparison: ValueComparison
    name_prefix: str = ''
    name_suffix: str = ''
    extra_attributes: dict = None
    custom_name: str | None = None

    def generate_kpis(self, dataset: DatasetComparison) -> list[KPI]:
        """
        Generate comparison KPIs from a DatasetComparison.

        Process:
        1. Get reference and variation datasets from comparison dataset
        2. Generate base KPIs for both datasets
        3. Match KPIs by object_name
        4. Apply comparison operation
        5. Create comparison KPI instances

        Args:
            dataset: DatasetComparison containing reference and variation datasets

        Returns:
            List of comparison KPI instances

        Raises:
            AttributeError: If dataset is not a DatasetComparison
        """
        if not isinstance(dataset, DatasetComparison):
            raise TypeError(
                f"ComparisonKPIDefinition requires a {DatasetComparison.__name__}, got {type(dataset).__name__}"
            )

        reference_dataset = dataset.reference_dataset
        variation_dataset = dataset.variation_dataset

        reference_kpis = self.base_definition.generate_kpis(reference_dataset)
        variation_kpis = self.base_definition.generate_kpis(variation_dataset)

        variation_kpis_by_object = {
            kpi.attributes.object_name: kpi
            for kpi in variation_kpis
        }

        comparison_kpis = []

        for ref_kpi in reference_kpis:
            obj_name = ref_kpi.attributes.object_name
            if obj_name not in variation_kpis_by_object:
                continue

            var_kpi = variation_kpis_by_object[obj_name]
            comparison_value = self.comparison(var_kpi.value, ref_kpi.value)

            # If custom_name is set and there are multiple objects, append object name
            kpi_custom_name = self.custom_name
            if self.custom_name and len(reference_kpis) > 1:
                kpi_custom_name = f"{self.custom_name} {obj_name}"

            unit = self.comparison.unit or ref_kpi.attributes.unit
            attributes = KPIAttributes(
                flag=ref_kpi.attributes.flag,
                model_flag=ref_kpi.attributes.model_flag,
                object_name=obj_name,
                aggregation=ref_kpi.attributes.aggregation,
                dataset_name=dataset.name,
                dataset_type=type(dataset),
                value_comparison=self.comparison,
                reference_dataset_name=reference_dataset.name,
                variation_dataset_name=variation_dataset.name,
                name_prefix=self.name_prefix,
                name_suffix=self.name_suffix,
                custom_name=kpi_custom_name,
                unit=unit,
                dataset_attributes=dataset.attributes,
                extra_attributes=self.extra_attributes or dict()
            )

            comparison_kpi = KPI(
                value=comparison_value,
                attributes=attributes,
                dataset=dataset
            )
            for k in [var_kpi, ref_kpi]:
                obj_info = k.get_object_info_from_model()
                if isinstance(obj_info, pd.Series) and not obj_info.empty:
                    comparison_kpi._object_info = obj_info
                    break

            comparison_kpis.append(comparison_kpi)

        return comparison_kpis

    def required_flags(self) -> set[FlagTypeProtocol]:
        """
        Return required flags from base definition.

        Returns:
            Set of flags required by base definition
        """
        return self.base_definition.required_flags()
