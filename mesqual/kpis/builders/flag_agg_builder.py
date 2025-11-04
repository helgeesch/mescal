from __future__ import annotations

from typing import Literal, Any, Hashable, Callable, TYPE_CHECKING
from dataclasses import dataclass
import pandas as pd

from mesqual.kpis.aggregations import Aggregation
from mesqual.units import Units
from mesqual.kpis.definitions.flag_aggregation import FlagAggKPIDefinition
from mesqual.kpis.builders.base import KPIBuilder

if TYPE_CHECKING:
    from mesqual.typevars import FlagTypeProtocol


@dataclass
class ModelPropertyFilter:
    """
    Configuration for filtering objects based on model properties.

    Filters a model DataFrame (with objects as index) by applying property constraints,
    query expressions, and custom filter functions. Returns the list of object names
    (index values) that pass all filters.

    All conditions are combined with AND logic.
    """
    properties: dict[str, Any] | None = None
    query_expr: str | None = None
    filter_funcs: dict[str, Callable[[Any], bool]] | None = None

    def apply_filter(self, model_df: pd.DataFrame) -> list[Hashable]:
        """
        Apply all filter conditions to model DataFrame and return matching object names.

        Args:
            model_df: DataFrame with objects as index and properties as columns

        Returns:
            List of object names (index values) that pass all filter conditions

        Process:
            1. Apply property filters (exact match or list membership) by row filtering
            2. Apply query expression using pandas.query()
            3. Apply custom filter functions column-wise
            4. Return index of remaining rows
        """
        model_dff = model_df.copy()
        if self.properties:
            for property, value in self.properties.items():
                if isinstance(value, (list, set)):
                    model_dff = model_dff[model_dff[property].isin(value)]
                else:
                    model_dff = model_dff[model_dff[property] == value]

        if self.query_expr:
            model_dff = model_dff.query(self.query_expr, engine="python")

        if self.filter_funcs:
            for prop_name, filter_func in self.filter_funcs.items():
                model_dff = model_dff[prop_name].apply(filter_func)

        return list(model_dff.index)


class FlagAggKPIBuilder(KPIBuilder[FlagAggKPIDefinition]):
    """
    Fluent builder for creating FlagAggregation KPI definitions in bulk.

    Provides a clean, declarative API for specifying multiple KPI definitions
    at once. Particularly useful for creating large numbers of KPIs with
    different combinations of flags and aggregations.

    Example:
        
        >>> builder = FlagAggKPIBuilder()
        >>> definitions = (
        ...     builder
        ...     .for_flags(['BZ.market_price', 'BZ.net_position'])
        ...     .with_aggregations([Aggregations.Mean, Aggregations.Max, Aggregations.Min])
        ...     .for_all_objects()
        ...     .build()
        ... )
        >>> len(definitions)  # 2 flags × 3 aggs = 6

    All methods return self for chaining.
    """

    def __init__(self):
        """Initialize builder with default values."""
        super().__init__()
        self._flags: list[FlagTypeProtocol] = []
        self._aggregations: list[Aggregation] = []
        self._objects: list[Hashable] | Literal['auto'] = 'auto'
        self._model_flags: dict[FlagTypeProtocol, FlagTypeProtocol] = {}

    def for_flag(self, flag: FlagTypeProtocol) -> FlagAggKPIBuilder:
        """
        Set a single flag.

        Args:
            flag: Flag (e.g., 'BZ.Results.market_price')

        Returns:
            Self for chaining
        """
        self._flags = [flag]
        return self

    def for_flags(self, flags: list[FlagTypeProtocol]) -> FlagAggKPIBuilder:
        """
        Set multiple flags.

        Args:
            flags: List of flags

        Returns:
            Self for chaining
        """
        self._flags = flags
        return self

    def with_aggregation(self, agg: Aggregation) -> FlagAggKPIBuilder:
        """
        Set a single aggregation.

        Args:
            agg: Aggregation function

        Returns:
            Self for chaining
        """
        self._aggregations = [agg]
        return self

    def with_aggregations(self, aggs: list[Aggregation]) -> FlagAggKPIBuilder:
        """
        Set multiple aggregations.

        Args:
            aggs: List of aggregation functions

        Returns:
            Self for chaining
        """
        self._aggregations = aggs
        return self

    def for_all_objects(self) -> FlagAggKPIBuilder:
        """
        Auto-discover objects from data.

        Objects will be detected from DataFrame columns when
        generate_kpis() is called.

        Returns:
            Self for chaining
        """
        self._objects = 'auto'
        return self

    def for_objects_with_model_properties(
            self,
            properties: dict[str, Any] | None = None,
            query_expr: str | None = None,
            filter_funcs: dict[str, Callable[[Any], bool]] | None = None
    ) -> 'FlagAggKPIBuilder':
        """
        Filter objects based on model properties during KPI generation.

        During KPI generation, the model DataFrame is fetched and filtered using the
        specified conditions. Only objects that:
        1. Pass all filter conditions, AND
        2. Exist in both model and flag DataFrames
        will be included in the generated KPIs.

        Three modes of operation (can be combined with AND logic):
            1. Property filters: Exact match or list membership
            2. Query expression: Pandas query string evaluated on model DataFrame
            3. Filter functions: Custom functions applied to property columns

        All conditions across all modes are combined with AND logic.

        Args:
            properties: Dict of property names to values. Scalars for exact match,
                lists/sets for membership checks.
            query_expr: Pandas query expression (uses engine="python")
            filter_funcs: Dict of property names to filter functions applied column-wise

        Returns:
            Self for chaining

        Examples:

            # Property filter - exact match and list membership
            builder.for_objects_with_model_properties(
                properties={'country': 'DE', 'type': ['wind', 'solar']}
            )

            # Query expression
            builder.for_objects_with_model_properties(
                query_expr='country == "DE" and voltage_kV > 200'
            )

            # Custom filter function
            builder.for_objects_with_model_properties(
                filter_funcs={'voltage_kV': lambda x: x > 200}
            )

            # Combined - properties AND query
            builder.for_objects_with_model_properties(
                properties={'country': 'DE'},
                query_expr='voltage_kV > 200'
            )
        """
        self._objects = ModelPropertyFilter(
            properties=properties,
            query_expr=query_expr,
            filter_funcs=filter_funcs
        )
        return self

    def for_objects(self, objects: list[Hashable]) -> FlagAggKPIBuilder:
        """
        Specify explicit object list.

        Args:
            objects: List of object names

        Returns:
            Self for chaining
        """
        self._objects = objects
        return self

    def with_model_flag(self, flag: FlagTypeProtocol, model_flag: FlagTypeProtocol) -> FlagAggKPIBuilder:
        """
        Set explicit model flag for a specific flag.

        Args:
            flag: Variable flag
            model_flag: Corresponding model flag

        Returns:
            Self for chaining
        """
        self._model_flags[flag] = model_flag
        return self

    def build(self) -> list[FlagAggKPIDefinition]:
        """
        Generate all KPI definitions from builder configuration.

        Creates the Cartesian product of flags × aggregations.

        Returns:
            List of FlagAggKPIDefinition instances

        Example:

            >>> builder = FlagAggKPIBuilder()
            >>> definitions = (
            ...     builder
            ...     .for_flags(['BZ.market_price', 'BZ.net_position'])
            ...     .with_aggregations([Aggregations.Mean, Aggregations.Max, Aggregations.Min])
            ...     .build()
            ... )
            >>> len(definitions)  # 2 flags × 3 aggs = 6
        """
        definitions = []

        for flag in self._flags:
            for agg in self._aggregations:
                definition = FlagAggKPIDefinition(
                    flag=flag,
                    aggregation=agg,
                    model_flag=self._model_flags.get(flag),
                    objects=self._objects,
                    name_prefix=self._name_prefix,
                    name_suffix=self._name_suffix,
                    custom_name=self._custom_name,
                    extra_attributes=self._extra_attributes,
                )
                definitions.append(definition)

        return definitions
