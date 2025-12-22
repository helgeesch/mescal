"""KPI Definitions - Specifications for KPI computation."""

from mesqual.kpis.definitions.base import KPIDefinition
from mesqual.kpis.definitions.flag_aggregation import FlagAggKPIDefinition
from mesqual.kpis.definitions.custom import CustomKPIDefinition
from mesqual.kpis.definitions.comparison import ComparisonKPIDefinition

__all__ = [
    'KPIDefinition',
    'FlagAggKPIDefinition',
    'CustomKPIDefinition',
    'ComparisonKPIDefinition',
]
