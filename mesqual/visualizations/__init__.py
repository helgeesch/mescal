"""MESQUAL Visualizations Package.

This package provides comprehensive visualization capabilities for energy systems analysis,
including interactive maps, time series dashboards, and data export functionality.

The visualizations package supports multiple output formats and interactive components
designed specifically for multi-scenario energy modeling analysis and comparison.

Modules:
    - value_mapping_system: Data value mapping and color scaling utilities
    - folium_viz_system: Interactive map visualization and Legend system for Folium maps

Classes:
    - TimeSeriesDashboardGenerator: Creates interactive Plotly time series dashboards
    - HTMLDashboard: Generates comprehensive HTML analysis dashboards
    - HTMLTable: Creates formatted HTML tables for data presentation

Example:

    >>> from mesqual.visualizations import HTMLDashboard, TimeSeriesDashboardGenerator
    >>> dashboard = HTMLDashboard()
    >>> ts_gen = TimeSeriesDashboardGenerator()
"""

from . import folium_viz_system
from . import folium_viz_system as folviz
from . import value_mapping_system
from . import value_mapping_system as valmap
from .plotly_figures.timeseries_dashboard import TimeSeriesDashboardGenerator
from .html_dashboard import HTMLDashboard
from .html_table import HTMLTable

__all__ = [
    'value_mapping_system',
    'valmap',
    'folium_viz_system',
    'folviz',
    'TimeSeriesDashboardGenerator',
    'HTMLDashboard',
    'HTMLTable',
]
