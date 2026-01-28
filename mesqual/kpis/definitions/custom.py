from __future__ import annotations

from abc import abstractmethod
from typing import Any, Literal, TYPE_CHECKING, Hashable, Optional
import pandas as pd

from mesqual.kpis.definitions.base import KPIDefinition
from mesqual.kpis.kpi import KPI
from mesqual.kpis.attributes import KPIAttributes
from mesqual.kpis.aggregations import Aggregation
from mesqual.units import Units
from mesqual.datasets.dataset import Dataset

if TYPE_CHECKING:
    from mesqual.typevars import FlagTypeProtocol


class CustomKPIDefinition(KPIDefinition):
    """
    Base class for custom KPI computation logic.

    Supports two computation patterns:
        1. Batch computation: Override compute_batch()
        2. Per-object computation: Override compute_for_object()

    Choose the pattern that best fits your use case:
        - Use batch for efficient vectorized operations across all objects
        - Use per_object for complex logic that varies significantly per object

    Examples:

        Volume Weighted Price per country implementation:
            >>> class VolumeWeightedPrice(CustomKPIDefinition):
            ...    def get_unit(self) -> Units.Unit:
            ...        return Units.EUR_per_MWh
            ...
            ...    def required_flags(self) -> set[FlagTypeProtocol]:
            ...        return {'countries_t.price', 'countries_t.load'}
            ...
            ...    def compute_batch(
            ...            self,
            ...            dataset: Dataset,
            ...            objects: list[Hashable] | Literal['auto_from_batch_generation']
            ...    ) -> dict[Hashable, Any]:
            ...        df_price = dataset.fetch('countries_t.price')
            ...        df_load = dataset.fetch('countries_t.load')
            ...
            ...        if isinstance(objects, list):
            ...            df_price = df_price[objects]
            ...            df_load = df_load[objects]
            ...
            ...        vol_weighted_price = (df_price * df_load).sum() / df_load.sum()
            ...        return vol_weighted_price.to_dict()
            ...
            >>> from mesqual import StudyManager
            >>> study: StudyManager
            >>> vol_weighted_price_definition = VolumeWeightedPrice('countries_t.vol_weighted_price')
            >>> study.scen.add_kpis_from_definitions_to_all_child_datasets([vol_weighted_price_definition])

    """

    def __init__(
        self,
        kpi_flag: FlagTypeProtocol,
        model_flag: Optional[FlagTypeProtocol] = None,
        objects: list[Hashable] | Literal['auto_from_model_flag', 'auto_from_batch_generation'] = 'auto_from_model_flag',
        name_prefix: str = '',
        name_suffix: str = '',
        extra_attributes: dict = None,
        aggregation: Aggregation | None = None
    ):
        """
        Initialize custom KPI definition.

        Args:
            kpi_flag: Variable flag for the KPI (Can be a hypothetical flag that doesn't exist
            model_flag: Optional model flag (auto-inferred if None)
            objects: List of objects or 'auto_from_model_flag' / 'auto_from_batch_generation' to discover
            name_prefix: Prefix for KPI names
            name_suffix: Suffix for KPI names
            aggregation: Optional aggregation (for metadata only, not used in computation)
        """
        self.kpi_flag = kpi_flag
        self.model_flag = model_flag
        self.objects = objects
        self.name_prefix = name_prefix
        self.name_suffix = name_suffix
        self.extra_attributes = extra_attributes
        self.aggregation = aggregation

    def generate_kpis(self, dataset: Dataset) -> list[KPI]:
        """
        Generate KPIs using either per-object or batch computation.

        Args:
            dataset: Dataset to compute KPIs for

        Returns:
            List of computed KPI instances
        """
        model_flag = self.model_flag or dataset.flag_index.get_linked_model_flag(self.kpi_flag)

        if self.objects == 'auto_from_model_flag':
            objects = dataset.fetch(model_flag).index.tolist()
        else:
            objects = self.objects

        try:
            return self._generate_kpis_batch(dataset, model_flag, objects)
        except NotImplementedError:
            pass

        if self.objects == 'auto_from_batch_generation':
            raise ValueError(f'If you want to use auto_from_batch_generation, you must override compute_batch().')

        try:
            return self._generate_kpis_per_object(dataset, model_flag, objects)
        except NotImplementedError:
            raise NotImplementedError("Must override compute_for_object() or compute_batch().")

    def _generate_kpis_per_object(
        self,
        dataset: Dataset,
        model_flag: FlagTypeProtocol,
        objects: list[Hashable]
    ) -> list[KPI]:
        """
        Generate KPIs by calling compute_for_object() for each object.

        Args:
            dataset: Dataset to compute for
            model_flag: Model flag for objects
            objects: List of object names

        Returns:
            List of KPI instances
        """
        kpis = []
        for obj in objects:
            value = self.compute_for_object(dataset, obj)

            attributes = self._build_attributes(obj, dataset, model_flag)

            kpi = KPI(
                value=value,
                attributes=attributes,
                dataset=dataset
            )
            kpis.append(kpi)

        return kpis

    def _generate_kpis_batch(
        self,
        dataset: Dataset,
        model_flag: FlagTypeProtocol,
        objects: list[Hashable] | Literal['auto_from_batch_generation']
    ) -> list[KPI]:
        """
        Generate KPIs by calling compute_batch() once.

        Args:
            dataset: Dataset to compute for
            model_flag: Model flag for objects
            objects: List of object names

        Returns:
            List of KPI instances
        """
        # Compute all values at once
        values_dict = self.compute_batch(dataset, objects)

        kpis = []
        for obj, value in values_dict.items():
            attributes = self._build_attributes(obj, dataset, model_flag)

            kpi = KPI(
                value=value,
                attributes=attributes,
                dataset=dataset
            )
            kpis.append(kpi)

        return kpis

    def compute_for_object(self, dataset: Dataset, object_name: Hashable) -> Any:
        """
        Compute KPI value for a single object.

        Override this for per-object computation pattern.

        Args:
            dataset: Dataset to fetch data from
            object_name: Name of the object to compute for

        Returns:
            Computed KPI value

        Raises:
            NotImplementedError: If not overridden
        """
        raise NotImplementedError("Must override compute_for_object() or compute_batch()")

    def compute_batch(
            self,
            dataset: Dataset,
            objects: list[Hashable] | Literal['auto_from_batch_generation']
    ) -> dict[Hashable, Any]:
        """
        Compute KPI values for all objects at once.

        Override this for batch computation pattern.

        Args:
            dataset: Dataset to fetch data from
            objects: List of object names to compute for

        Returns:
            Dict mapping object_name → value

        Raises:
            NotImplementedError: If not overridden
        """
        raise NotImplementedError("Must override compute_for_object() or compute_batch()")

    @abstractmethod
    def get_unit(self) -> Units.Unit:
        """
        Return the unit for this KPI type.

        Returns:
            Physical unit for the KPI values
        """
        pass

    def _build_attributes(
        self,
        object_name: Hashable,
        dataset: Dataset,
        model_flag: FlagTypeProtocol
    ) -> KPIAttributes:
        """
        Build KPIAttributes for a KPI instance.

        Args:
            object_name: Object identifier
            dataset: Source dataset
            model_flag: Model flag for the object

        Returns:
            KPIAttributes instance
        """

        return KPIAttributes(
            flag=self.kpi_flag,
            model_flag=model_flag,
            object_name=object_name,
            aggregation=self.aggregation,
            dataset_name=dataset.name,
            dataset_type=type(dataset),
            name_prefix=self.name_prefix,
            name_suffix=self.name_suffix,
            unit=self.get_unit(),
            dataset_attributes=dataset.attributes,
            extra_attributes=self.extra_attributes or dict()
        )
