"""
MESQUAL KPI System V2 (kpis)

A high-performance, attribute-rich KPI system for energy systems analysis.

When to Use the KPI System vs Direct DataFrame Processing:
----------------------------------------------------------
The KPI framework is NOT required for basic metric extraction. For quick analysis,
you can simply fetch time-series data and process it directly with pandas:

    >>> # Quick approach: Direct dataframe processing
    >>> df = study.scen.fetch('BiddingZone.Results.market_price')
    >>> mean_price = df.mean().unstack('dataset')  # Fast and straightforward

However, the KPI system provides unique advantages for complex workflows:

    1. **Model Object Integration**: KPIs retain links to model object metadata,
       enabling property-based filtering and queries (e.g., filter KPIs by bus
       voltage level, generator carrier type, or custom model properties)

    2. **Folium Visualization Integration**: KPIs work seamlessly with MESQUAL's
       map visualization system, automatically providing geographic context and
       tooltips with related metrics

    3. **Related KPI Discovery**: Within a KPICollection, easily find related metrics
       (e.g., same object/scenario but different aggregation, time period, or unit)

    4. **Advanced Unit Handling**: Automatic unit conversion and validation when
       working with collections of heterogeneous KPIs (critical for multi-physics
       analysis with mixed energy/power/price metrics)

    5. **Multi-Scenario Workflows**: Built-in support for comparison KPIs and
       bulk computation across scenario collections with progress tracking

Use the KPI system when you need these features; use direct dataframe processing
for quick, one-off calculations that stay within the pandas framework.

Core Components:

    - KPI: Single computed metric with rich metadata
    - KPIAttributes: Metadata container for filtering and grouping
    - KPICollection: Container with advanced filtering and export
    - KPIDefinition: Abstract base API for KPI specifications
    - KPIBuilder: Abstract base API for creating KPIDefinitions in bulk

Architecture and High Level Workflow:

    KPIBuilder (instructions on how to create a set of definitions)
        ↓
    KPIDefinition (what to compute)
        ↓
    KPI (computed result + metadata)
        ↓
    KPICollection (filtering, querying, visualization)

Definitions:

    - FlagAggKPIDefinition: Standard flag + aggregation KPIs (most common)
    - CustomKPIDefinition: Abstract base for study-specific computation logic
    - ComparisonKPIDefinition: Delta calculations between scenarios

Aggregations:

    - Aggregations: Standard aggregation functions (Mean, Sum, Max, etc.)
    - ValueComparisons: Comparison operations (Increase, PercentageIncrease, etc.)
    - ArithmeticValueOperations: Arithmetic operations (Product, Division, etc.)

Example Usage - Standard KPIs:

    >>> from mesqual.kpis import (
    ...     KPI, KPICollection, FlagAggKPIDefinition, Aggregations
    ... )
    >>>
    >>> # Create definition for standard aggregation
    >>> definition = FlagAggKPIDefinition(
    ...     flag='BZ.Results.market_price',
    ...     aggregation=Aggregations.Mean
    ... )
    >>>
    >>> # Generate and add KPIs to a dataset
    >>> dataset: Dataset
    >>> dataset.add_kpis_from_definitions(definition)
    >>>
    >>> # Access collection with filtering and unit handling
    >>> collection = dataset.kpi_collection
    >>> german_kpis = collection.filter_by_model_properties(properties={'country': 'DE'})
    >>> df = german_kpis.to_dataframe(unit_handling='auto_convert')

Example Usage - Custom KPIs:

For study-specific calculations, subclass CustomKPIDefinition and implement
either compute_for_object() or compute_batch() plus get_unit():

    >>> from mesqual.kpis import CustomKPIDefinition
    >>> from mesqual.units import Units
    >>>
    >>> class GeneratorCapacityFactor(CustomKPIDefinition):
    ...     '''Custom KPI calculating system efficiency per generator.'''
    ...
    ...     def compute_for_object(self, dataset, object_name):
    ...         # Fetch data for specific generator
    ...         generation = dataset.fetch('generators_t.p')[object_name].sum()
    ...         capacity = dataset.fetch('generators').loc[object_name, 'p_nom']
    ...         return (generation / (capacity * 8760)) * 100
    ...
    ...     def get_unit(self):
    ...         return Units.percent
    >>>
    >>> # Use like standard definitions
    >>> custom_def = SystemEfficiencyKPI(
    ...     flag='generators_t.p',
    ...     aggregation=None  # Optional metadata
    ... )
    >>> study.scen.add_kpis_from_definitions_to_all_child_datasets(custom_def)

Note: For many custom metrics, it's often cleaner to first create a study-specific
flag (variable) via a custom flag interpreter, then apply standard FlagAggKPIDefinition
aggregations to that flag. This keeps computation logic in the interpreter layer
and enables you to fetch the flag also as a time-series or outside of the KPI framework.
"""

# Core classes
from mesqual.kpis.kpi import KPI
from mesqual.kpis.attributes import KPIAttributes
from mesqual.kpis.collection import KPICollection

# Definitions
from mesqual.kpis.definitions.base import KPIDefinition
from mesqual.kpis.definitions.flag_aggregation import FlagAggKPIDefinition
from mesqual.kpis.definitions.custom import CustomKPIDefinition
from mesqual.kpis.definitions.comparison import ComparisonKPIDefinition

# Aggregations
from mesqual.kpis.aggregations import (
    Aggregation,
    Aggregations,
    ValueComparison,
    ValueComparisons,
    ArithmeticValueOperation,
    ArithmeticValueOperations,
)

# Builders
from mesqual.kpis.builders.flag_agg_builder import FlagAggKPIBuilder
from mesqual.kpis.builders.comparison_builder import ComparisonKPIBuilder

__all__ = [
    # Core
    'KPI',
    'KPIAttributes',
    'KPICollection',

    # Definitions
    'KPIDefinition',
    'FlagAggKPIDefinition',
    'CustomKPIDefinition',
    'ComparisonKPIDefinition',

    # Aggregations
    'Aggregation',
    'Aggregations',
    'ValueComparison',
    'ValueComparisons',
    'ArithmeticValueOperation',
    'ArithmeticValueOperations',

    # Builders
    'FlagAggKPIBuilder',
    'ComparisonKPIBuilder',
]
