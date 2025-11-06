from dataclasses import dataclass
from typing import Callable, Type, Any

from shapely import Point
import folium

from mesqual.visualizations.folium_viz_system.visualizable_data_item import VisualizableDataItem, KPIDataItem

from mesqual.visualizations.folium_viz_system.base_viz_system import (
    ResolvedFeature,
    FeatureResolver,
    PropertyMapper,
    FoliumObjectGenerator,
)


@dataclass
class ResolvedTextOverlayFeature(ResolvedFeature):
    """
    Resolved visual properties for text overlay elements.
    
    Container for all computed styling properties of text overlay visualizations,
    including font styling, positioning, colors, and shadow effects.
    Used by TextOverlayGenerator to create folium markers with styled text content.
    """

    @property
    def location(self) -> Point:
        return self.get('location')

    @property
    def offset(self) -> tuple[float, float]:
        return self.get('offset')

    @property
    def azimuth_angle(self) -> float:
        return self.get('azimuth_angle')

    @property
    def text_color(self) -> str:
        return self.get('text_color')

    @property
    def font_size(self) -> str:
        return self.get('font_size')

    @property
    def font_weight(self) -> str:
        return self.get('font_weight')

    @property
    def background_color(self) -> str:
        return self.get('background_color')

    @property
    def shadow_size(self) -> float:
        return self.get('shadow_size')

    @property
    def shadow_color(self) -> str:
        return self.get('shadow_color')


class TextOverlayFeatureResolver(FeatureResolver[ResolvedTextOverlayFeature]):
    """
    Resolves visual properties for text overlay elements.
    
    Specialized feature resolver for text overlay visualizations that handles text
    content, font styling, positioning, and visual effects. Commonly used for
    adding data labels, value displays, or annotations to map elements.
    
    Args:
        text_color: Text color (static value or PropertyMapper)
        font_size: Font size with units like '10pt' (static value or PropertyMapper)
        font_weight: Font weight like 'bold', 'normal' (static value or PropertyMapper)
        background_color: Background color for text (static value or PropertyMapper)
        shadow_size: Text shadow size like '0.5px' (static value or PropertyMapper)
        shadow_color: Text shadow color (static value or PropertyMapper)
        text_print_content: Text content to display (True for auto-generated)
        tooltip: Tooltip content (True for auto-generated, False for none)
        popup: Popup content (True/False/PropertyMapper)
        location: Point location (defaults to smart location detection)
        **property_mappers: Additional custom property mappings
        
    Examples:
        Basic value labels:
        >>> resolver = TextOverlayFeatureResolver(
        ...     text_color='#000000',
        ...     font_size='12pt',
        ...     font_weight='bold'
        ... )
        
        Data-driven text styling:
        >>> resolver = TextOverlayFeatureResolver(
        ...     text_color=PropertyMapper.from_kpi_value(
        ...         lambda v: '#FF0000' if v > 100 else '#000000'
        ...     ),
        ...     font_size=PropertyMapper.from_kpi_value(
        ...         lambda v: f'{min(max(8 + v/50, 8), 20)}pt'
        ...     ),
        ...     text_print_content=PropertyMapper.from_kpi_value(
        ...         lambda v: f'{v:.0f} MW'
        ...     )
        ... )
    """
    def __init__(
            self,
            text_color: PropertyMapper | str = '#3A3A3A',
            font_size: PropertyMapper | str = '10pt',
            font_weight: PropertyMapper | str = 'bold',
            background_color: PropertyMapper | str = None,
            shadow_size: PropertyMapper | str = '0.5px',
            shadow_color: PropertyMapper | str = '#F2F2F2',
            text_print_content: PropertyMapper | str | bool = True,
            tooltip: PropertyMapper | str | bool = True,
            popup: PropertyMapper | folium.Popup | bool = False,
            location: PropertyMapper | Point = None,
            offset: PropertyMapper | tuple[float, float] = (0, 0),
            azimuth_angle: PropertyMapper | float = 90,
            **property_mappers: PropertyMapper | Any,
    ):
        mappers = dict(
            text_color=text_color,
            font_size=font_size,
            font_weight=font_weight,
            background_color=background_color,
            shadow_size=shadow_size,
            shadow_color=shadow_color,
            text_print_content=text_print_content,
            tooltip=tooltip,
            popup=popup,
            location=self._explicit_or_fallback(location, self._default_location_mapper()),
            offset=offset,
            azimuth_angle=azimuth_angle,
            **property_mappers
        )
        super().__init__(feature_type=ResolvedTextOverlayFeature, **mappers)


class TextOverlayGenerator(FoliumObjectGenerator[TextOverlayFeatureResolver]):
    """
    Generates text overlays for map data items.
    
    Creates folium Marker objects with styled HTML text content overlaid on
    the map. Handles text formatting, positioning, shadow effects, and
    responsive styling based on data values.
    
    Commonly used for displaying:
    - KPI values directly on map elements (power flows, prices, etc.)
    - Data labels for areas, lines, or points
    - Dynamic text that changes based on underlying data
    - Status indicators or categorical labels
    
    Examples:
        Value display on bidding zones:
        >>> from mesqual.units import Units
        >>> text_gen = TextOverlayGenerator(
        ...     TextOverlayFeatureResolver(
        ...         text_print_content=PropertyMapper(
        ...             lambda di: Units.get_pretty_text_for_quantity(
        ...                 di.kpi.quantity, decimals=0, include_unit=False
        ...             )
        ...         ),
        ...         font_size='10pt',
        ...         font_weight='bold'
        ...     )
        ... )
        >>> 
        >>> fg = folium.FeatureGroup('Price Labels')
        >>> text_gen.generate_objects_for_kpi_collection(price_kpis, fg)
        >>> fg.add_to(map)
        
        Combined with area visualization:
        >>> area_gen = AreaGenerator(...)
        >>> text_gen = TextOverlayGenerator(...)
        >>> 
        >>> area_gen.generate_objects_for_model_df(zones_df, feature_group)
        >>> text_gen.generate_objects_for_model_df(zones_df, feature_group)
    """
    """Generates text overlays for map data items."""

    def _feature_resolver_type(self) -> Type[TextOverlayFeatureResolver]:
        return TextOverlayFeatureResolver

    def generate(self, data_item: VisualizableDataItem, feature_group: folium.FeatureGroup) -> None:
        """
        Generate and add a folium Marker with styled text overlay.
        
        Args:
            data_item: Data item containing point location and text content
            feature_group: Folium feature group to add the text marker to
        """
        style = self.feature_resolver.resolve_feature(data_item)
        if not isinstance(style.location, Point):
            return

        if not style.text_print_content:
            return

        text_content = style.text_print_content
        text_color = style.text_color
        font_size = style.font_size
        font_weight = style.font_weight
        shadow_size = style.shadow_size
        shadow_color = style.shadow_color

        angle = float(style.azimuth_angle) - 90  # Folium counts clockwise from right-pointing direction; normal convention is CCW
        x_offset, y_offset = style.offset

        icon_html = f'''
            <div style="
                position: absolute;
                left: 50%;
                top: 50%;
                transform: translate(-50%, -50%) translate({x_offset}px, {-y_offset}px) rotate({angle}deg);
                text-align: center;
                font-size: {font_size};
                font-weight: {font_weight};
                color: {text_color};
                white-space: nowrap;
                text-shadow:
                   -{shadow_size} -{shadow_size} 0 {shadow_color},  
                    {shadow_size} -{shadow_size} 0 {shadow_color},
                   -{shadow_size}  {shadow_size} 0 {shadow_color},
                    {shadow_size}  {shadow_size} 0 {shadow_color};
            ">
                {text_content}
            </div>
        '''

        folium.Marker(
            location=(style.location.y, style.location.x),
            icon=folium.DivIcon(html=icon_html),
            tooltip=style.tooltip,
            popup=style.popup,
        ).add_to(feature_group)

    def _get_contrasting_color(self, surface_color: str) -> str:
        """Get contrasting text color for a surface color."""
        if self._is_dark(surface_color):
            return '#F2F2F2'
        return '#3A3A3A'

    def _get_shadow_color(self, text_color: str) -> str:
        """Get shadow color for text."""
        if text_color == '#F2F2F2':
            return '#3A3A3A'
        return '#F2F2F2'

    @staticmethod
    def _is_dark(color: str) -> bool:
        """Check if a color is dark."""
        if not color.startswith('#'):
            return False
        try:
            r, g, b = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
            return (0.299 * r + 0.587 * g + 0.114 * b) < 160
        except (ValueError, IndexError):
            return False


if __name__ == '__main__':
    import os
    import webbrowser
    import pandas as pd
    from shapely.geometry import Polygon
    import folium

    from mesqual.visualizations.value_mapping_system import (
        SegmentedContinuousColorscale,
        SegmentedContinuousOpacityMapping,
    )
    from mesqual.visualizations.folium_viz_system.viz_areas import AreaGenerator, AreaFeatureResolver

    area_df = pd.DataFrame({
        'geometry': [
            Polygon([(7.0, 50.0), (7.5, 50.0), (7.5, 50.5), (7.0, 50.5)]),
            Polygon([(8.0, 50.0), (8.5, 50.0), (8.5, 50.5), (8.0, 50.5)]),
            Polygon([(9.0, 50.0), (9.5, 50.0), (9.5, 50.5), (9.0, 50.5)])
        ],
        'value': [10, 20, 30]
    }, index=['area1', 'area2', 'area3'])

    m = folium.Map(location=[50.25, 8.0], zoom_start=8, tiles='CartoDB Positron')

    color_map = SegmentedContinuousColorscale.single_segment_autoscale_factory_from_array(
        values=area_df['value'].values,
        colorscale=['green', 'blue', 'red']
    )

    opacity_map = SegmentedContinuousOpacityMapping.single_segment_autoscale_factory_from_array(
        values=area_df['value'].values,
        output_range=(0.4, 0.9)
    )

    area_generator = AreaGenerator(
        feature_resolver=AreaFeatureResolver(
            fill_color=PropertyMapper.from_item_attr('value', color_map),
            fill_opacity=PropertyMapper.from_item_attr('value', opacity_map),
            border_color='#ABABAB',
            border_width=10,
            tooltip=True,
        )
    )

    text_generator = TextOverlayGenerator()

    fg = folium.FeatureGroup(name='Test Areas')
    area_generator.generate_objects_for_model_df(area_df, fg)
    text_generator.generate_objects_for_model_df(area_df, fg)
    fg.add_to(m)

    m.add_child(folium.LayerControl())
    m.save('_tmp/map.html')
    webbrowser.open('file://' + os.path.abspath('_tmp/map.html'))
