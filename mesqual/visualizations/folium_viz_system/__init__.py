"""Folium-based visualization system for MESQUAL energy system analysis.

This module provides a comprehensive visualization framework built on Folium for creating
interactive maps and geospatial visualizations of energy system data. The system supports
multiple visualization types including areas, lines, markers, icons, and text overlays,
with automatic feature resolution and generation capabilities.

The visualization system follows a modular architecture with:
- Base visualization components for property mapping and feature resolution
- Specialized visualizers for KPI collections and data management
- Multiple visualization feature types (areas, lines, markers, overlays)
- Data item abstractions for model and KPI data integration

Example:

    Basic usage for creating area visualizations:
    >>> from mesqual.visualizations.folium_viz_system import AreaGenerator, AreaFeatureResolver
    >>> area_resolver = AreaFeatureResolver(property_mapper)
    >>> area_generator = AreaGenerator()
    >>> area_features = area_resolver.resolve_features(data_items)
    >>> folium_map = area_generator.add_features_to_map(folium_map, area_features)

Attributes:
    The module exports visualization components, feature resolvers, and data abstractions
    for building interactive energy system maps and analysis dashboards.
"""

from .base_viz_system import PropertyMapper
from .kpi_collection_map_visualizer import KPICollectionMapVisualizer, KPIGroupingManager
from .visualizable_data_item import VisualizableDataItem, ModelDataItem, KPIDataItem
from .viz_areas import ResolvedAreaFeature, AreaFeatureResolver, AreaGenerator
from .viz_arrow_icon import ResolvedArrowIconFeature, ArrowIconFeatureResolver, ArrowIconGenerator
from .viz_circle_marker import ResolvedCircleMarkerFeature, CircleMarkerFeatureResolver, CircleMarkerGenerator
# from .viz_icons import ResolvedIconFeature, IconFeatureResolver, IconGenerator
from .viz_line_text_overlay import ResolvedLineTextOverlayFeature, LineTextOverlayFeatureResolver, LineTextOverlayGenerator
from .viz_lines import ResolvedLineFeature, LineFeatureResolver, LineGenerator
from .viz_text_overlay import ResolvedTextOverlayFeature, TextOverlayFeatureResolver, TextOverlayGenerator

from . import legends

__all__ = [
    # Base visualization components
    "PropertyMapper",

    # KPI visualization and management
    "KPICollectionMapVisualizer",
    "KPIGroupingManager",

    # Data item abstractions
    "VisualizableDataItem",
    "ModelDataItem",
    "KPIDataItem",

    # Area visualization
    "ResolvedAreaFeature",
    "AreaFeatureResolver",
    "AreaGenerator",

    # Arrow icon visualization
    "ResolvedArrowIconFeature",
    "ArrowIconFeatureResolver",
    "ArrowIconGenerator",

    # Circle marker visualization
    "ResolvedCircleMarkerFeature",
    "CircleMarkerFeatureResolver",
    "CircleMarkerGenerator",

    # Line text overlay visualization
    "ResolvedLineTextOverlayFeature",
    "LineTextOverlayFeatureResolver",
    "LineTextOverlayGenerator",

    # Line visualization
    "ResolvedLineFeature",
    "LineFeatureResolver",
    "LineGenerator",

    # Text overlay visualization
    "ResolvedTextOverlayFeature",
    "TextOverlayFeatureResolver",
    "TextOverlayGenerator",

    "legends"
]
