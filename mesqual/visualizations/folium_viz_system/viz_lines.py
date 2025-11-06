import hashlib
from dataclasses import dataclass
from typing import List, Type, Any

import folium
from folium.plugins import AntPath, PolyLineOffset
from shapely import LineString, MultiLineString

from mesqual.visualizations.folium_viz_system.visualizable_data_item import VisualizableDataItem

from mesqual.visualizations.folium_viz_system.base_viz_system import (
    ResolvedFeature,
    FeatureResolver,
    PropertyMapper,
    FoliumObjectGenerator,
)
from mesqual.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ResolvedLineFeature(ResolvedFeature):
    """
    Resolved visual properties for line/connection map elements.
    
    Container for all computed styling properties of line visualizations,
    including colors, widths, dash patterns, and animation effects.
    Used by LineGenerator to create folium PolyLine objects.
    """
    """Specialized style container for line visualizations."""

    @property
    def geometry(self) -> LineString:
        return self.get('geometry')

    @property
    def line_color(self) -> str:
        return self.get('line_color')

    @property
    def line_width(self) -> float:
        return max(self.get('line_width'), 0.0)

    @property
    def line_opacity(self) -> float:
        return max(min(self.get('line_opacity'), 1.0), 0.0)

    @property
    def dash_pattern(self) -> List[int]:
        return self.get('dash_pattern')

    @property
    def reverse_path_direction(self) -> bool:
        return self.get('reverse_path_direction')

    @property
    def line_ant_path(self) -> bool:
        return self.get('line_ant_path')

    @property
    def line_ant_path_delay(self) -> int:
        return self.get('line_ant_path_delay')

    @property
    def line_ant_path_pulse_color(self) -> str:
        return self.get('line_ant_path_pulse_color')


class LineFeatureResolver(FeatureResolver[ResolvedLineFeature]):
    """
    Resolves visual properties for line/connection map elements.
    
    Specialized feature resolver for line visualizations that handles LineString
    geometries, styling options like colors and widths, and advanced features
    like animated ant paths. Commonly used for visualizing transmission lines,
    interconnectors, or any linear connections between geographic points.
    
    Args:
        line_color: Line stroke color (static value or PropertyMapper)
        line_width: Line width in pixels (static value or PropertyMapper)
        line_opacity: Line transparency 0-1 (static value or PropertyMapper)
        dash_pattern: Dash pattern as list of integers (static value or PropertyMapper)
        line_ant_path: Enable animated "marching ants" effect (static value or PropertyMapper)
        line_ant_path_delay: Animation delay in milliseconds (static value or PropertyMapper)
        line_ant_path_pulse_color: Color of animation pulses (static value or PropertyMapper)
        reverse_path_direction: Reverse line direction (static value or PropertyMapper)
        tooltip: Tooltip content (True for auto-generated, False for none)
        popup: Popup content (True/False/PropertyMapper)
        geometry: LineString geometry (defaults to 'geometry' attribute)
        **property_mappers: Additional custom property mappings
        
    Examples:
        Basic line visualization:
        >>> resolver = LineFeatureResolver(
        ...     line_color='#FF0000',
        ...     line_width=3.0,
        ...     line_opacity=0.8
        ... )
        
        Data-driven flow visualization:
        >>> flow_color_scale = SegmentedContinuousColorscale(...)
        >>> width_scale = SegmentedContinuousLineWidthMapping(...)
        >>> resolver = LineFeatureResolver(
        ...     line_color=PropertyMapper.from_kpi_value(flow_color_scale),
        ...     line_width=PropertyMapper.from_kpi_value(width_scale),
        ...     line_ant_path=PropertyMapper.from_kpi_value(lambda v: abs(v) > 100)
        ... )
    """
    def __init__(
            self,
            line_color: PropertyMapper | str = '#000000',
            line_width: PropertyMapper | float = 3.0,
            line_opacity: PropertyMapper | float = 1.0,
            dash_pattern: PropertyMapper | List[int] = None,
            line_ant_path: PropertyMapper | bool = False,
            line_ant_path_delay: PropertyMapper | int = 1500,
            line_ant_path_pulse_color: PropertyMapper | str = '#DBDBDB',
            reverse_path_direction: PropertyMapper | bool = False,
            tooltip: PropertyMapper | str | bool = True,
            popup: PropertyMapper | folium.Popup | bool = False,
            geometry: PropertyMapper | LineString = None,
            **property_mappers: PropertyMapper | Any,
    ):
        mappers = dict(
            line_color=line_color,
            line_width=line_width,
            line_opacity=line_opacity,
            dash_pattern=dash_pattern,
            line_ant_path=line_ant_path,
            line_ant_path_delay=line_ant_path_delay,
            line_ant_path_pulse_color=line_ant_path_pulse_color,
            reverse_path_direction=reverse_path_direction,
            tooltip=tooltip,
            popup=popup,
            geometry=self._explicit_or_fallback(geometry, self._default_line_string_mapper()),
            **property_mappers
        )
        super().__init__(feature_type=ResolvedLineFeature, **mappers)


class LineGenerator(FoliumObjectGenerator[LineFeatureResolver]):
    """
    Generates folium PolyLine objects for line visualizations with offset management.
    
    Creates interactive map lines from data items with computed styling properties.
    Handles LineString geometries, applies styling, and manages automatic offset
    positioning to prevent overlapping lines on the same route.
    
    Features automatic line offset registry to handle multiple lines between
    the same points, animated ant paths for flow visualization, and support
    for dash patterns and opacity effects.
    
    Args:
        feature_resolver: LineFeatureResolver for computing visual properties
        per_feature_group_offset_registry: Use separate offset tracking per feature group
        offset_increment: Pixel increment for line offsets (default: 5)
    
    Commonly used for visualizing:
    - Transmission lines colored by power flows
    - Interconnectors with bidirectional flow indicators
    - Transport connections with capacity or utilization data
    - Network links with performance metrics
    
    Examples:
        Basic flow visualization:
        >>> color_scale = SegmentedContinuousColorscale(...)
        >>> width_scale = SegmentedContinuousLineWidthMapping(...)
        >>> generator = LineGenerator(
        ...     LineFeatureResolver(
        ...         line_color=PropertyMapper.from_kpi_value(color_scale),
        ...         line_width=PropertyMapper.from_kpi_value(width_scale),
        ...         tooltip=True
        ...     ),
        ...     offset_increment=10  # Larger offset for visibility
        ... )
        >>> 
        >>> fg = folium.FeatureGroup('Power Flows')
        >>> generator.generate_objects_for_kpi_collection(flow_kpis, fg)
        >>> fg.add_to(map)
        
        Animated flow indication:
        >>> resolver = LineFeatureResolver(
        ...     line_ant_path=PropertyMapper.from_kpi_value(lambda v: abs(v) > 500),
        ...     line_ant_path_delay=2000,
        ...     reverse_path_direction=PropertyMapper.from_kpi_value(lambda v: v < 0)
        ... )
        >>> generator = LineGenerator(resolver)
    """

    def __init__(
            self,
            feature_resolver: LineFeatureResolver = None,
            per_feature_group_offset_registry: bool = True,
            offset_increment: int = 5,
    ):
        super().__init__(feature_resolver)
        self.offset_increment = offset_increment
        self.per_feature_group_registry = per_feature_group_offset_registry
        self._global_registry: dict[str, int] = {}
        self._group_registry: dict[int, dict[str, int]] = {}

    def reset_registry(self) -> None:
        self._global_registry.clear()
        self._group_registry.clear()

    def _feature_resolver_type(self) -> Type[LineFeatureResolver]:
        return LineFeatureResolver

    def generate(self, data_item: VisualizableDataItem, feature_group: folium.FeatureGroup) -> None:
        """
        Generate and add a folium PolyLine to the feature group with automatic offset.
        
        Args:
            data_item: Data item containing LineString geometry and associated data
            feature_group: Folium feature group to add the line to
        """
        style = self.feature_resolver.resolve_feature(data_item)
        geometry = style.geometry
        if not isinstance(geometry, (LineString, MultiLineString)):
            return

        coordinates = [(lat, lon) for lon, lat in geometry.coords]
        effective_offset = self._get_line_offset(coordinates, feature_group)

        if style.reverse_path_direction:
            coordinates = coordinates[::-1]

        line_kwargs = dict(
            locations=coordinates,
            color=style.line_color,
            weight=style.line_width,
            opacity=style.line_opacity,
            offset=effective_offset,
            tooltip=style.tooltip,
            dash_array=style.dash_pattern or None,
            popup=style.popup,
        )

        if style.line_ant_path:
            if effective_offset not in [None, 0, 0.0]:
                logger.warning(
                    f'Trying to set an offset with an animated LineAntPath is not possible. '
                    f'Offset will not be applied for {data_item.get_name()}.'
                )
            line_kwargs.update(dict(
                paused=False,
                reverse=False,
                hardware_acceleration=False,
                delay=style.line_ant_path_delay,
                pulse_color=style.line_ant_path_pulse_color,
            ))
            poly_line = AntPath(**line_kwargs)
        else:
            poly_line = PolyLineOffset(**line_kwargs)

        poly_line.add_to(feature_group)

    def _get_line_offset(self, coordinates: list[tuple[float, float]], feature_group: folium.FeatureGroup) -> float:
        line_hash = self._hash_coordinates(coordinates)
        registry = self._get_registry_for_group(feature_group)
        offset_index = registry.get(line_hash, 0)
        registry[line_hash] = offset_index + 1
        offset_px = offset_index * self.offset_increment
        offset_side = 1 if offset_index % 2 == 0 else -1
        effective_offset = offset_px * offset_side
        return effective_offset

    def _get_registry_for_group(self, feature_group: folium.FeatureGroup) -> dict[str, int]:
        if not self.per_feature_group_registry:
            return self._global_registry
        group_id = id(feature_group)
        if group_id not in self._group_registry:
            self._group_registry[group_id] = {}
        return self._group_registry[group_id]

    def _hash_coordinates(self, coordinates: list[tuple[float, float]]) -> str:
        rounded = [(round(lat, 6), round(lon, 6)) for lat, lon in coordinates]
        return hashlib.md5(str(rounded).encode("utf-8")).hexdigest()


if __name__ == '__main__':
    import os
    import webbrowser
    import pandas as pd
    from shapely.geometry import Polygon, LineString
    import folium

    from mesqual.visualizations.value_mapping_system import (
        SegmentedContinuousColorscale,
        SegmentedContinuousOpacityMapping,
    )

    line_df = pd.DataFrame({
        'geometry': [
            LineString([(7.0, 50.0), (7.5, 52.0)]),
            LineString([(8.0, 50.0), (8.5, 52.0)]),
            LineString([(9.0, 50.0), (9.5, 52.0)]),
            LineString([(9.0, 50.0), (9.5, 52.0)]),
        ],
        'flow': [10, 20, 30, 29]
    }, index=['line1', 'line2', 'line3.1', 'line3.2'])

    m = folium.Map(location=[50.25, 8.0], zoom_start=7, tiles='CartoDB Positron')

    color_map = SegmentedContinuousColorscale.single_segment_autoscale_factory_from_array(
        values=line_df['flow'].values,
        colorscale=['green', 'blue', 'red']
    )

    opacity_map = SegmentedContinuousOpacityMapping.single_segment_autoscale_factory_from_array(
        values=line_df['flow'].values,
        output_range=(0.4, 0.9)
    )

    line_generator = LineGenerator(
        feature_resolver=LineFeatureResolver(
            line_color=PropertyMapper.from_item_attr('flow', color_map),
            line_opacity=PropertyMapper.from_item_attr('flow', opacity_map),
            line_width=10,
            tooltip=False,
            popup=True,
        ),
        offset_increment=15,
    )

    fg = folium.FeatureGroup(name='Test Lines')
    line_generator.generate_objects_for_model_df(line_df, fg)
    fg.add_to(m)

    m.add_child(folium.LayerControl())
    m.save('_tmp/map.html')
    webbrowser.open('file://' + os.path.abspath('_tmp/map.html'))
