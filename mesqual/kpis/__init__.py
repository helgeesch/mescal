"""
MESQUAL KPI System V2 (kpis)

A high-performance, attribute-rich KPI system for energy systems analysis.

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
    - FlagAggregationDefinition: Simple flag + aggregation KPIs
    - CustomKPIDefinition: Custom computation logic

Aggregations:
    - Aggregations: Standard aggregation functions (Mean, Sum, Max, etc.)
    - ValueComparisons: Comparison operations (Increase, PercentageIncrease, etc.)
    - ArithmeticValueOperations: Arithmetic operations (Product, Division, etc.)

Example Usage:

    >>> from mesqual.kpis import (
    ...     KPI, KPICollection, FlagAggKPIDefinition, Aggregations
    ... )
    >>>
    >>> # Create definition
    >>> definition = FlagAggKPIDefinition(
    ...     flag='BZ.Results.market_price',
    ...     aggregation=Aggregations.Mean
    ... )
    >>>
    >>> # Generate and add KPIs to a dataset
    >>> dataset: Dataset
    >>> dataset.add_kpis_from_definitions(definition)
    >>>
    >>> # Add to collection
    >>> collection = dataset.kpi_collection
    >>>
    >>> # Filter and export
    >>> german_kpis = collection.filter_by_model_properties(properties={'country': 'DE'})
    >>> df = german_kpis.to_dataframe(unit_handling='auto_convert')
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
