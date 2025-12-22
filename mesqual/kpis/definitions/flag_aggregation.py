from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING, Hashable
import pandas as pd

from mesqual.kpis.definitions.base import KPIDefinition
from mesqual.kpis.kpi import KPI
from mesqual.kpis.attributes import KPIAttributes
from mesqual.kpis.aggregations import Aggregation
from mesqual.units import Units
from mesqual.datasets.dataset import Dataset

if TYPE_CHECKING:
    from mesqual.typevars import FlagTypeProtocol
    from mesqual.kpis.builders.flag_agg_builder import ModelPropertyFilter


@dataclass
class FlagAggKPIDefinition(KPIDefinition):
    """
    Simple flag + aggregation KPI definition.

    Computes KPIs by:
    1. Fetching flag data (DataFrame with objects as columns)
    2. Applying aggregation (vectorized across all columns)
    3. Creating one KPI per object (column)

    This is the most common KPI type for energy system analysis.
    """

    flag: FlagTypeProtocol
    aggregation: Aggregation
    model_flag: FlagTypeProtocol | None = None  # Auto-inferred if None
    objects: list[Hashable] | Literal['auto'] | ModelPropertyFilter = 'auto'  # 'auto' = discover from data
    extra_attributes: dict = None
    name_prefix: str = ''
    name_suffix: str = ''
    custom_name: str | None = None  # Complete name override

    def generate_kpis(self, dataset: Dataset) -> list[KPI]:
        """
        Generate KPIs by batch computation.

        Process:
        1. Infer model_flag if not provided
        2. Fetch data (DataFrame)
        3. Discover objects from DataFrame columns if objects='auto'
        4. Apply aggregation to entire DataFrame (vectorized)
        5. Create KPI instance per object with metadata

        Args:
            dataset: Dataset to compute KPIs for

        Returns:
            List of computed KPI instances
        """
        from mesqual.kpis.builders.flag_agg_builder import ModelPropertyFilter

        model_flag = self.model_flag or dataset.flag_index.get_linked_model_flag(self.flag)

        df = dataset.fetch(self.flag)

        model_df = dataset.fetch(model_flag) if dataset.flag_is_accepted(model_flag) else None
        if self.objects == 'auto':
            objects = df.columns.tolist()
        elif isinstance(self.objects, ModelPropertyFilter):
            if model_df is None:
                raise Exception(f'No Model DF found for model_flag "{model_flag}" in variable_flag "{self.flag}"')
            filtered_model_df_objects = self.objects.apply_filter(model_df)
            objects = [o for o in df.columns if o in filtered_model_df_objects]
        else:
            objects = self.objects

        aggregated = self.aggregation(df)  # Returns Series with one value per column

        dataset_type = type(dataset)

        unit = dataset.flag_index.get_unit(self.flag)
        if self.aggregation.unit is not None:
            unit = self.aggregation.unit

        kpis = []
        for obj in objects:
            if obj not in aggregated.index:
                # TODO: optional warning if object listed but not present in flag
                continue  # Skip objects not in aggregated results

            # If custom_name is set and there are multiple objects, append object name
            kpi_custom_name = self.custom_name
            if self.custom_name and len(objects) > 1:
                kpi_custom_name = f"{self.custom_name} {obj}"

            attributes = KPIAttributes(
                flag=self.flag,
                model_flag=model_flag,
                object_name=obj,
                aggregation=self.aggregation,
                dataset_name=dataset.name,
                dataset_type=dataset_type,
                name_prefix=self.name_prefix,
                name_suffix=self.name_suffix,
                custom_name=kpi_custom_name,
                unit=unit,
                dataset_attributes=dataset.attributes,
                extra_attributes=self.extra_attributes or dict()
            )

            kpi = KPI(
                value=aggregated[obj],
                attributes=attributes,
                dataset=dataset
            )
            if (model_df is not None) and (obj in model_df.index):
                kpi._object_info = model_df.loc[obj]
            kpis.append(kpi)

        return kpis

    def required_flags(self) -> set[FlagTypeProtocol]:
        """
        Return required flags.

        Returns:
            Set of required flags
        """
        flags = {self.flag}
        if self.model_flag:
            flags.add(self.model_flag)
        return flags
