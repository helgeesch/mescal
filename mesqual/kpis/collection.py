from __future__ import annotations

from typing import Any, Callable, Literal, Iterable
from collections import defaultdict
import pandas as pd

from mesqual.kpis.kpi import KPI
from mesqual.units import Units


class KPICollection:
    """
    Collection of KPIs with filtering, grouping, and export capabilities.

    Provides rich query interface for finding related KPIs and organizing
    KPIs for visualization and reporting.
    """

    def __init__(self, kpis: Iterable[KPI] | None = None):
        """
        Initialize KPI collection.

        Args:
            kpis: Optional list of KPIs to initialize with
        """
        self._kpis: list[KPI] = list(kpis) if kpis is not None else []

    def add(self, kpi: KPI) -> None:
        """Add single KPI to collection."""
        self._kpis.append(kpi)

    def extend(self, kpis: list[KPI]) -> None:
        """Add multiple KPIs to collection."""
        self._kpis.extend(kpis)

    def clear(self) -> None:
        """Remove all KPIs from collection."""
        self._kpis.clear()

    def filter(self, **attribute_filters) -> KPICollection:
        """
        Filter KPIs by exact attribute matches.

        Args:
            **attribute_filters: Attribute name-value pairs to filter by

        Returns:
            New KPICollection with filtered KPIs

        Examples:

            collection.filter(flag='BZ.Results.market_price')
            collection.filter(object_name='DE-LU', aggregation=Aggregations.Mean)
            collection.filter(dataset_type='scenario')
        """
        filtered = []
        for kpi in self._kpis:
            primitive = kpi.to_dict(primitive_values=True)
            non_primitive = kpi.to_dict(primitive_values=False)
            match = all(
                primitive.get(attr, None) == value
                or non_primitive.get(attr, None) == value
                for attr, value in attribute_filters.items()
            )
            if match:
                filtered.append(kpi)
        return KPICollection(filtered)

    def filter_by_model_properties(
            self,
            properties: dict[str, Any] | None = None,
            query_expr: str | None = None,
            filter_funcs: dict[str, Callable[[Any], bool]] | None = None
    ) -> KPICollection:
        """
        Filter KPIs by properties from their model objects.

        Three modes of operation (can be combined with AND logic):
            1. Property filters: Exact match or list membership
            2. Query expression: Pandas query string
            3. Filter functions: Custom functions applied to properties

        All conditions across all modes are combined with AND logic.

        Args:
            properties: Dict of property names to values. Scalars for exact match,
                lists for membership checks.
            query_expr: Pandas query expression (uses engine="python")
            filter_funcs: Dict of property names to filter functions

        Returns:
            New KPICollection with filtered KPIs

        Examples:

            # Property filter - exact match and list membership
            collection.filter_by_model_properties(
                properties={'country': 'DE', 'type': ['wind', 'solar']}
            )

            # Query expression
            collection.filter_by_model_properties(
                query_expr='country == "DE" and voltage_kV > 200'
            )

            # Custom filter function
            collection.filter_by_model_properties(
                filter_funcs={'voltage_kV': lambda x: x > 200}
            )

            # Combined - properties AND query
            collection.filter_by_model_properties(
                properties={'country': 'DE'},
                query_expr='voltage_kV > 200'
            )

            # Combined - properties AND filter functions
            collection.filter_by_model_properties(
                properties={'type': ['wind', 'solar']},
                filter_funcs={'voltage_kV': lambda x: x > 200}
            )
        """
        filtered = []

        for kpi in self._kpis:
            if kpi.attributes.object_name is None:
                continue

            obj_info = kpi.get_object_info_from_model()

            if self._series_passes_property_filters(obj_info, properties, query_expr, filter_funcs):
                filtered.append(kpi)

        return KPICollection(filtered)

    def filter_by_kpi_attributes(
            self,
            attributes: dict[str, Any] | None = None,
            query_expr: str | None = None,
            filter_funcs: dict[str, Callable[[Any], bool]] | None = None
    ) -> KPICollection:
        """
        Filter KPIs by their attributes.

        Three modes of operation (can be combined with AND logic):
            1. Property filters: Exact match or list membership
            2. Query expression: Pandas query string
            3. Filter functions: Custom functions applied to attributes

        All conditions across all modes are combined with AND logic.

        Args:
            attributes: Dict of attribute names to values. Scalars for exact match,
                lists for membership checks.
            query_expr: Pandas query expression (uses engine="python")
            filter_funcs: Dict of attribute names to filter functions

        Returns:
            New KPICollection with filtered KPIs

        Examples:

            # Property filter - exact match and list membership
            collection.filter_by_kpi_attributes(
                attributes={'flag': 'BZ.Results.price'}
            )

            # Query expression
            collection.filter_by_kpi_attributes(
                query_expr='flag.str.contains("price") and value > 100'
            )

            # Custom filter function
            collection.filter_by_kpi_attributes(
                filter_funcs={'value': lambda x: x > 100}
            )

            # Combined - properties AND query
            collection.filter_by_kpi_attributes(
                attributes={'flag': 'BZ.Results.price'},
                query_expr='value > 100'
            )
        """
        filtered = []

        for kpi in self._kpis:
            kpi_attrs = pd.Series(kpi.to_dict(primitive_values=True))

            if self._series_passes_property_filters(kpi_attrs, attributes, query_expr, filter_funcs):
                filtered.append(kpi)

        return KPICollection(filtered)

    @staticmethod
    def _series_passes_property_filters(
            data: pd.Series,
            properties: dict[str, Any] | None = None,
            query_expr: str | None = None,
            filter_funcs: dict[str, Callable[[Any], bool]] | None = None
    ) -> bool:
        """
        Apply property filters to a data Series.

        Private method used by both filter_by_model_properties and filter_by_kpi_attributes.

        Args:
            data: Series containing property/attribute data
            properties: Dict of property names to values for exact match or list membership
            query_expr: Pandas query expression
            filter_funcs: Dict of property names to filter functions

        Returns:
            True if data passes all filters, False otherwise
        """
        if properties is not None:
            for prop_name, expected_value in properties.items():
                if prop_name not in data.index:
                    return False

                actual_value = data[prop_name]

                if isinstance(expected_value, list):
                    if actual_value not in expected_value:
                        return False
                else:
                    if actual_value != expected_value:
                        return False

        if query_expr is not None:
            temp_df = pd.DataFrame({k: [v] for k, v in data.items()})
            try:
                result = temp_df.query(query_expr, engine="python")
                if result.empty:
                    return False
            except:
                return False

        if filter_funcs is not None:
            for prop_name, filter_function in filter_funcs.items():
                if prop_name not in data.index:
                    return False

                try:
                    if not filter_function(data[prop_name]):
                        return False
                except:
                    return False

        return True

    def group_by(self, *attributes: str) -> dict[tuple, KPICollection]:
        """
        Group KPIs by attribute values.

        Args:
            *attributes: Attribute names to group by

        Returns:
            Dictionary mapping attribute value tuples to KPICollections

        Example:

            >>> groups = collection.group_by('flag', 'aggregation')
                {('BZ.Results.market_price', Aggregations.Mean): KPICollection(...), ...}
        """
        groups = defaultdict(list)

        for kpi in self._kpis:
            key = tuple(
                getattr(kpi.attributes, attr, None)
                or kpi.attributes.dataset_attributes.get(attr)
                for attr in attributes
            )
            groups[key].append(kpi)

        return {k: KPICollection(v) for k, v in groups.items()}

    def get_related(
        self,
        reference_kpi: KPI,
        vary_attributes: list[str],
        exclude_attributes: list[str] | None = None
    ) -> 'KPICollection':
        """
        Find KPIs related to reference, varying only specified attributes.

        Args:
            reference_kpi: The KPI to find relatives for
            vary_attributes: Attributes that can differ (e.g., ['aggregation'])
            exclude_attributes: Attributes to ignore in comparison (e.g., ['name_prefix'])

        Returns:
            New KPICollection with related KPIs

        Examples:

            # Same object/flag, different aggregations
            collection.get_related(my_kpi, vary_attributes=['aggregation'])

            # Same object/flag/agg, different datasets
            collection.get_related(my_kpi, vary_attributes=['dataset_name'])

            # Same flag/agg, different objects
            collection.get_related(my_kpi, vary_attributes=['object_name'])
        """
        exclude_attributes = exclude_attributes or ['name_prefix', 'name_suffix', 'custom_name']
        ref_attrs = reference_kpi.attributes.as_dict()
        related = []

        for kpi in self._kpis:
            if kpi is reference_kpi:
                continue

            kpi_attrs = kpi.attributes.as_dict()
            match = True

            for attr_name, ref_value in ref_attrs.items():
                # Skip attributes we want to vary or exclude
                if attr_name in vary_attributes or attr_name in exclude_attributes:
                    continue

                if kpi_attrs.get(attr_name) != ref_value:
                    match = False
                    break

            if match:
                related.append(kpi)

        return KPICollection(related)

    def get_all_attribute_values(self, attribute: str) -> set:
        """
        Get all unique values for a specific attribute across collection.

        Args:
            attribute: Attribute name to get values for

        Returns:
            Set of unique values for the attribute
        """
        values = set()
        for kpi in self._kpis:
            val = getattr(kpi.attributes, attribute, None)
            if val is None:
                val = kpi.attributes.dataset_attributes.get(attribute)
            if val is not None:
                values.add(val)
        return values

    def get_all_kpi_attributes_and_value_sets(self, primitive_values: bool = False) -> dict[str, set]:
        """
        Get all attribute names and their unique value sets.

        Used by KPIGroupingManager for intelligent grouping.

        Args:
            primitive_values: If True, convert values to primitives

        Returns:
            Dictionary mapping attribute names to sets of values
        """
        attribute_sets = defaultdict(set)

        for kpi in self._kpis:
            attrs = kpi.attributes.as_dict(primitive_values=primitive_values)
            for attr_name, attr_value in attrs.items():
                if attr_value is not None:
                    attribute_sets[attr_name].add(attr_value)

        return dict(attribute_sets)

    def get_in_common_kpi_attributes(self, primitive_values: bool = False) -> dict:
        """
        Get attributes that have same value across all KPIs in collection.

        Args:
            primitive_values: If True, convert values to primitives

        Returns:
            Dictionary of common attributes
        """
        if self.empty:
            return {}

        # Start with first KPI's attributes
        common = self._kpis[0].attributes.as_dict(primitive_values=primitive_values)

        # Remove any that differ in subsequent KPIs
        for kpi in self._kpis[1:]:
            attrs = kpi.attributes.as_dict(primitive_values=primitive_values)
            common = {
                k: v for k, v in common.items()
                if attrs.get(k) == v
            }

        return common

    def to_dataframe(
        self,
        unit_handling: Literal['original', 'auto_convert', 'target', 'custom'] = 'original',
        target_unit: Units.Unit | None = None,
        target_units_by_group: dict[tuple, Units.Unit] | None = None,
        group_by_attributes: list[str] | None = None,
        normalize_to_collection: bool = False
    ) -> pd.DataFrame:
        """
        Export KPIs as DataFrame with flexible unit handling.

        Args:
            unit_handling: Strategy for unit conversion
                - 'original': Keep original units
                - 'auto_convert': Convert to pretty units per KPI
                - 'target': Use single target_unit for all
                - 'custom': Use target_units_by_group mapping
            target_unit: Single target unit (when unit_handling='target')
            target_units_by_group: Dict mapping group key → target unit
            group_by_attributes: Attributes to group by for 'custom' mode
            normalize_to_collection: If True, find common "pretty" unit across entire collection

        Returns:
            DataFrame with KPI data

        Examples:
            # Keep original units
            df = collection.to_dataframe()

            # Auto-convert each KPI to its own pretty unit
            df = collection.to_dataframe(unit_handling='auto_convert')

            # Convert all to single unit
            df = collection.to_dataframe(
                unit_handling='target',
                target_unit=Units.MEUR
            )

            # Custom units per group
            df = collection.to_dataframe(
                unit_handling='custom',
                group_by_attributes=['flag', 'aggregation'],
                target_units_by_group={
                    ('consumer_surplus', 'Sum'): Units.BEUR,
                    ('producer_surplus', 'Sum'): Units.MEUR,
                }
            )

            # Normalize to common pretty unit for collection
            df = collection.to_dataframe(normalize_to_collection=True)
        """
        data = []

        # Normalize to collection: find common pretty unit
        common_unit = None
        if normalize_to_collection:
            quantities = [kpi.quantity for kpi in self._kpis]
            try:
                common_unit = Units.get_common_pretty_unit_for_quantities(quantities)
            except ValueError:
                # Quantities have different dimensionalities, fall back to auto_convert
                pass

        for kpi in self._kpis:
            quantity = kpi.quantity

            # Apply unit conversion based on strategy
            if normalize_to_collection and common_unit:
                quantity = quantity.to(common_unit)

            elif unit_handling == 'auto_convert':
                quantity = Units.get_quantity_in_pretty_unit(quantity)

            elif unit_handling == 'target' and target_unit:
                quantity = quantity.to(target_unit)

            elif unit_handling == 'custom' and target_units_by_group and group_by_attributes:
                # Build group key
                kpi_attributes = kpi.attributes.as_dict(primitive_values=True)
                group_key = tuple(
                    kpi_attributes[attr] if attr in kpi_attributes else None
                    for attr in group_by_attributes
                )
                if group_key in target_units_by_group:
                    quantity = quantity.to(target_units_by_group[group_key])

            row = {
                'name': kpi.name,
                **kpi.attributes.as_dict(primitive_values=True),
                'value': quantity.magnitude,
                'quantity': quantity,
                'unit': str(quantity.units),  # Set after attributes to avoid being overwritten
            }
            data.append(row)

        return pd.DataFrame(data)

    @property
    def kpis(self) -> dict[str, KPI]:
        return {kpi.name: kpi for kpi in self._kpis}

    def __iter__(self):
        """Iterate over KPIs in collection."""
        return iter(self._kpis)

    def __len__(self):
        """Number of KPIs in collection."""
        return len(self._kpis)

    def __getitem__(self, index):
        """Get KPI by index."""
        return self._kpis[index]

    def __contains__(self, item):
        kpi_name = item.name
        return any(kpi.name == kpi_name for kpi in self._kpis)

    @property
    def empty(self) -> bool:
        """Check if collection is empty."""
        return len(self._kpis) == 0

    @property
    def size(self) -> int:
        """Get number of KPIs in collection."""
        return len(self._kpis)

    def __repr__(self) -> str:
        """String representation of collection."""
        return f"KPICollection({self.size} KPIs)"
